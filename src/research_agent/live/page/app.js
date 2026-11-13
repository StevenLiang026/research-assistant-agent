"use strict";
// The chat page of scripts/ask.py --web (docs/m10-计划.md 5). The server keeps the state (live/web.py) and
// this script only shows it: every text goes in through textContent, and the only HTML set is what the
// server rendered for an answer or a note (setServerHtml).

const MAX_FOLLOW_UPS = 2;
const STATUS = {
  answered: "已回答", clarification: "追问", dead_loop: "循环中止", step_limit: "步数用尽",
  budget_stop: "预算用尽", invalid_output: "输出无效", context_overflow: "上下文超长", harness_error: "运行出错",
};
const PHASE = {idle: "空闲", running: "运行中", ending: "正在结束", ended: "已结束"};
const REFUSED = {
  busy: "上一轮还没有结束。", ending: "运行正在结束,不再接受新消息。", empty: "请求是空的。",
  too_long: "请求太长:最多 4000 字。", no_cookie: "页面没有权限:请打开控制台里打印的链接。",
  command: "以 / 开头或单独的 exit、quit 是控制台的命令,不会发给模型。换问题:直接输入请求"
    + "(追问待答时先点「新问题」);结束:点「结束运行」。",
  not_text: "消息里有无法编码的字符(多半是粘贴时截断的表情符号):请删掉后再发送。",
};

const view = {
  status: {phase: "idle", pending: null},
  exchanges: new Map(), // id -> exchange, in order
  elements: new Map(), // id -> the exchange's element
  opened: new Set(), // the step details left open: "exchange:step:attempt"
  awaiting: null, // the exchange this page sent last, until its turn is done
  source: null,
  ended: false,
};

function byId(id) {
  return document.getElementById(id);
}

function make(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

// The one place HTML is set: what the server rendered (live/render.py), an answer or a note.
function setServerHtml(element, html) {
  element.innerHTML = html;
}

function seconds(value) {
  return typeof value === "number" ? `${value.toFixed(1)} 秒` : "";
}

function pretty(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function compact(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}

// ---- the header -------------------------------------------------------------------------------------------

function showSystem(system) {
  const box = byId("system");
  box.replaceChildren();
  const mode = system.vote_mode ? `(${system.vote_mode})` : "";
  const votes = system.votes == null ? null : `${system.votes} 票${mode}`;
  const items = [
    ["系统", system.model],
    ["投票", votes],
    ["思考预算", system.reasoning_budget == null ? null : `${system.reasoning_budget} tok`],
    ["规则", system.rules_version],
    ["llama.cpp", system.build],
    ["冻结记录", system.frozen_sha256 ? `${system.frozen_sha256.slice(0, 12)}…` : null, system.frozen_sha256],
    ["服务器", system.server],
    ["运行目录", system.run],
  ];
  for (const [label, value, full] of items) {
    if (value == null || value === "") continue;
    const item = make("span", "item");
    item.append(make("span", "label", label), make("span", "value", value));
    if (full) item.title = full;
    box.append(item);
  }
}

function setLink(text, broken) {
  const link = byId("link");
  link.textContent = text;
  link.classList.toggle("broken", Boolean(broken));
}

function lastExchange() {
  let last = null;
  for (const exchange of view.exchanges.values()) last = exchange;
  return last;
}

function showStatus() {
  const awaited = view.awaiting === null ? undefined : view.exchanges.get(view.awaiting);
  const waiting = view.awaiting !== null && !(awaited && awaited.state === "done");
  let phase = view.ended ? "ended" : view.status.phase;
  if (phase === "idle" && waiting) phase = "running"; // sent, and its status may not have come yet
  const idle = phase === "idle";
  byId("phase").textContent = PHASE[phase] || phase;
  byId("input").disabled = !idle;
  byId("send").disabled = !idle;
  byId("new").disabled = !idle;
  byId("quit").disabled = phase === "ending" || phase === "ended";
  byId("busy").hidden = phase !== "running";
  const pending = idle ? view.status.pending : null;
  byId("pending").hidden = !pending;
  if (pending) {
    byId("pending-text").textContent = `下一条消息是对追问的回答 (${pending.follow_ups + 1}/${MAX_FOLLOW_UPS})`;
  }
  const last = lastExchange();
  let banner = "";
  if (phase === "ended") banner = "运行已结束。";
  else if (phase === "ending") banner = "运行正在结束:正在运行的这一轮结束后停止,不再接受新消息。";
  else if (idle && last && last.result && last.result.after === "restate") {
    banner = `已经补充过 ${MAX_FOLLOW_UPS} 次:请把完整的请求重新写一遍再发送。`;
  }
  byId("banner").textContent = banner;
  byId("banner").hidden = !banner;
}

// ---- an exchange ------------------------------------------------------------------------------------------

function callLabel(call) {
  const args = call.arguments;
  const named = args && typeof args === "object" && typeof args.arxiv_id === "string";
  return named ? `${call.name}(${args.arxiv_id})` : call.name;
}

function stepSummary(step) {
  const parts = [`第 ${step.step} 步` + (step.attempt > 1 ? `(第 ${step.attempt} 次尝试)` : "")];
  if (step.state === "running") {
    parts.push("进行中…");
  } else if (step.state === "interrupted") {
    parts.push("已中断");
  } else if (step.state === "error") {
    parts.push(`模型出错 ${step.error}`, seconds(step.seconds));
  } else {
    if (step.votes) parts.push(`票 ${step.votes.won == null ? "-" : step.votes.won}/${step.votes.k}`);
    if (step.thinking) parts.push(`思考 ${step.thinking.tokens} tok`);
    if (step.calls.length) parts.push(`调用 ${step.calls.map(callLabel).join(", ")}`);
    else if (step.answer_chars != null) parts.push(`回答 ${step.answer_chars} 字`);
    parts.push(seconds(step.seconds));
  }
  return parts.filter(Boolean).join(" · ");
}

function stepBody(step) {
  const body = make("div", "step-body");
  if (step.state === "running") {
    body.append(make("p", "muted", "模型还在生成这一步的回答。"));
    return body;
  }
  if (step.state === "interrupted") {
    body.append(make("p", "muted", "运行在这一步完成之前结束了。"));
    return body;
  }
  if (step.state === "error") body.append(make("p", "warn", `模型这一步出错:${step.error}`));
  if (step.votes) {
    const greedy = step.votes.won == null || step.votes.representative === "greedy";
    const from = greedy ? "贪心候选(没有样本投出票)" : `样本 #${step.votes.representative}`;
    const failed = step.votes.failed ? `;${step.votes.failed} 个候选失败` : "";
    body.append(make("p", "", `代表回答来自${from}${failed}`));
  }
  if (step.thinking && step.thinking.forced) body.append(make("p", "warn", "思考在预算处被强制收尾"));
  if (step.rules.length) {
    body.append(make("p", "warn", "规则改写了回答,下面的思考对应改写前的回答"));
    for (const change of step.rules) {
      const rule = make("div", "rule");
      rule.append(make("p", "label", `规则 ${change.rule}:改写前`), make("pre", "", pretty(change.before)),
        make("p", "label", "改写后"), make("pre", "", pretty(change.after)));
      body.append(rule);
    }
  }
  for (const call of step.calls) {
    body.append(make("p", "call", `调用 ${call.name} ${compact(call.arguments)}`));
  }
  if (step.state === "error") return body;
  body.append(make("p", "label", "思考"));
  body.append(make("pre", "reasoning", step.reasoning == null ? "(这一步没有思考文本)" : step.reasoning));
  return body;
}

function toolElement(tool) {
  const parts = [tool.name];
  if (tool.state === "running") parts.push("进行中…");
  else if (tool.state === "interrupted") parts.push("已中断");
  else if (tool.state === "error") parts.push(`出错 ${tool.error}`);
  else parts.push(tool.summary);
  parts.push(seconds(tool.seconds));
  if (typeof tool.waited === "number" && tool.waited > 0) parts.push(`其中等待限速 ${tool.waited.toFixed(1)} 秒`);
  return make("li", `tool ${tool.state}`, parts.filter(Boolean).join(" · "));
}

function stepElement(exchange, step) {
  const box = make("div", `step ${step.state}`);
  const details = make("details", "");
  const key = `${exchange.id}:${step.step}:${step.attempt}`;
  details.open = view.opened.has(key);
  details.addEventListener("toggle", () => {
    if (details.open) view.opened.add(key);
    else view.opened.delete(key);
  });
  details.append(make("summary", "", stepSummary(step)));
  details.append(stepBody(step));
  box.append(details);
  if (step.tools.length) {
    const tools = make("ul", "tools");
    for (const tool of step.tools) tools.append(toolElement(tool));
    box.append(tools);
  }
  return box;
}

function resultElement(result) {
  const box = make("div", `bubble assistant ${result.status || "error"}`);
  const status = result.status ? STATUS[result.status] || result.status : "出错";
  const head = [status, seconds(result.seconds), result.turn ? `第 ${result.turn} 轮` : "", result.today];
  box.append(make("p", "result-head", head.filter(Boolean).join(" · ")));
  if (result.error) box.append(make("pre", "error-text", result.error));
  if (result.html) {
    const answer = make("div", "answer");
    setServerHtml(answer, result.html);
    box.append(answer);
  }
  if (result.notes.length) {
    const chips = make("div", "chips");
    for (const note of result.notes) {
      const chip = make("button", "chip", `笔记 ${note.note_id}`);
      chip.type = "button";
      chip.addEventListener("click", () => openNote(note.note_id));
      chips.append(chip);
    }
    box.append(chips);
  }
  if (result.cited_unseen.length) {
    box.append(make("p", "warn", `引用了本轮工具输出里没有出现过的编号:${result.cited_unseen.join(", ")}`));
  }
  if (result.after === "answer") box.append(make("p", "hint", "这是追问:下一条消息会作为它的回答,接在原请求后面发送。"));
  return box;
}

function exchangeElement(exchange) {
  const box = make("article", "exchange");
  const user = make("div", "bubble user");
  if (exchange.follow_up > 0) user.append(make("p", "tag", `对追问的回答 ${exchange.follow_up}/${MAX_FOLLOW_UPS}`));
  user.append(make("p", "request", exchange.request));
  box.append(user);
  if (exchange.sent !== exchange.request) {
    const sent = make("p", "sent");
    sent.append(make("span", "label", "实际发给模型:"), make("span", "", exchange.sent));
    box.append(sent);
  }
  if (exchange.state === "queued") box.append(make("p", "muted", "排队中…"));
  if (exchange.steps.length) {
    const steps = make("div", "steps");
    for (const step of exchange.steps) steps.append(stepElement(exchange, step));
    box.append(steps);
  }
  if (exchange.state === "interrupted") box.append(make("p", "warn", "已中断:运行在这一轮完成之前结束了。"));
  if (exchange.result) box.append(resultElement(exchange.result));
  return box;
}

function nearBottom() {
  const list = byId("exchanges");
  return list.scrollHeight - list.scrollTop - list.clientHeight < 80;
}

function scrollToBottom() {
  const list = byId("exchanges");
  list.scrollTop = list.scrollHeight;
}

function showExchange(exchange) {
  if (!exchange || typeof exchange.id !== "number") return;
  const atBottom = nearBottom();
  view.exchanges.set(exchange.id, exchange);
  const element = exchangeElement(exchange);
  const old = view.elements.get(exchange.id);
  if (old) old.replaceWith(element);
  else byId("exchanges").append(element);
  view.elements.set(exchange.id, element);
  byId("empty").hidden = true;
  if (atBottom) scrollToBottom();
  showStatus();
}

function showSnapshot(snapshot) {
  showSystem(snapshot.system || {});
  view.status = snapshot.status;
  for (const element of view.elements.values()) element.remove();
  view.exchanges = new Map();
  view.elements = new Map();
  for (const exchange of snapshot.exchanges) showExchange(exchange);
  byId("empty").hidden = view.exchanges.size > 0;
  scrollToBottom();
  showStatus();
}

// ---- the notes --------------------------------------------------------------------------------------------

function noteElement(note) {
  const item = make("li", "note");
  const title = make("button", "note-title", note.title || note.note_id);
  title.type = "button";
  title.addEventListener("click", () => openNote(note.note_id));
  const meta = make("p", "note-meta");
  const link = make("a", "", note.arxiv_id);
  link.href = note.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  meta.append(link, make("span", "", ` · ${note.created}`));
  if (note.tags.length) meta.append(make("span", "tags", ` · ${note.tags.join(", ")}`));
  item.append(title, meta);
  return item;
}

async function loadNotes() {
  const list = byId("notes-list");
  const state = byId("notes-state");
  try {
    const reply = await fetch("/api/notes?limit=50");
    if (!reply.ok) throw new Error(`HTTP ${reply.status}`);
    const body = await reply.json();
    list.replaceChildren(...body.notes.map(noteElement));
    state.textContent = body.notes.length ? `最近的 ${body.notes.length} 条` : "笔记库里还没有笔记。";
  } catch (error) {
    state.textContent = "笔记读取失败。";
  }
}

async function openNote(noteId) {
  const viewer = byId("viewer");
  byId("viewer-box").hidden = false;
  byId("viewer-title").textContent = noteId;
  viewer.replaceChildren(make("p", "muted", "读取中…"));
  try {
    const reply = await fetch(`/api/notes/${encodeURIComponent(noteId)}`);
    if (reply.status === 404) {
      viewer.replaceChildren(make("p", "warn", "笔记库里没有这条笔记。"));
      return;
    }
    if (!reply.ok) throw new Error(`HTTP ${reply.status}`);
    const body = await reply.json();
    setServerHtml(viewer, body.html);
  } catch (error) {
    viewer.replaceChildren(make("p", "warn", "笔记读取失败。"));
  }
}

// ---- talking to the server --------------------------------------------------------------------------------

async function post(path, payload) {
  const reply = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload || {}),
  });
  let body = null;
  try {
    body = await reply.json();
  } catch (error) {
    body = null;
  }
  return {status: reply.status, body};
}

function showError(text) {
  byId("error").textContent = text;
  byId("error").hidden = !text;
}

function refusal(reply) {
  const word = reply.body && reply.body.error;
  return REFUSED[word] || `没有发出(HTTP ${reply.status})。`;
}

async function send() {
  const input = byId("input");
  if (!input.value.trim() || input.disabled) return;
  input.disabled = true;
  byId("send").disabled = true;
  try {
    const reply = await post("/api/ask", {text: input.value});
    if (reply.status === 202) {
      input.value = "";
      showError("");
      const exchange = reply.body.exchange;
      view.awaiting = exchange.id; // until its turn is done, which the stream may have said already
      if (!view.exchanges.has(exchange.id)) showExchange(exchange);
      scrollToBottom(); // the message just sent is shown, and the steps that follow it stay in view
    } else {
      showError(refusal(reply));
    }
  } catch (error) {
    showError("没有发出:连接断开,可能已按 Ctrl+C。");
  }
  showStatus();
  if (!input.disabled) input.focus();
}

async function newQuestion() {
  try {
    const reply = await post("/api/new");
    if (reply.status === 200) {
      view.status = reply.body.status;
      showError("");
    } else {
      showError(refusal(reply));
    }
  } catch (error) {
    showError("连接断开,可能已按 Ctrl+C。");
  }
  showStatus();
}

async function quit() {
  if (!window.confirm("结束这次运行?正在运行的一轮会先跑完。")) return;
  try {
    const reply = await post("/api/quit");
    if (reply.status !== 202) showError(refusal(reply));
  } catch (error) {
    showError("连接断开,可能已按 Ctrl+C。");
  }
}

function connect(since) {
  const source = new EventSource(`/api/events?since=${since}`);
  view.source = source;
  source.addEventListener("open", () => setLink("已连接"));
  source.addEventListener("error", () => {
    if (!view.ended) setLink("连接断开,可能已按 Ctrl+C", true);
  });
  source.addEventListener("snapshot", (event) => showSnapshot(JSON.parse(event.data)));
  source.addEventListener("exchange", (event) => showExchange(JSON.parse(event.data)));
  source.addEventListener("status", (event) => {
    view.status = JSON.parse(event.data);
    showStatus();
  });
  source.addEventListener("notes", () => loadNotes());
  source.addEventListener("bye", (event) => {
    source.close();
    view.ended = true;
    const reason = JSON.parse(event.data).reason;
    setLink(reason === "interrupted" ? "运行已结束(Ctrl+C)" : "运行已结束");
    showStatus();
  });
}

function onKey(event) {
  if (event.key !== "Enter") return;
  if (!event.isComposing && event.keyCode !== 229) { // else the IME is confirming a word: the Enter is its own
    event.preventDefault(); // a message is one line: Enter never adds a line break
    send();
  }
}

async function start() {
  byId("send").addEventListener("click", send);
  byId("input").addEventListener("keydown", onKey);
  byId("new").addEventListener("click", newQuestion);
  byId("quit").addEventListener("click", quit);
  byId("notes-refresh").addEventListener("click", loadNotes);
  byId("viewer-close").addEventListener("click", () => {
    byId("viewer-box").hidden = true;
  });
  loadNotes();
  try {
    const reply = await fetch("/api/state");
    if (!reply.ok) {
      const forbidden = reply.status === 403;
      setLink(forbidden ? "页面没有权限:请打开控制台里打印的链接" : `读取失败(HTTP ${reply.status})`, true);
      return;
    }
    const snapshot = await reply.json();
    showSnapshot(snapshot);
    connect(snapshot.seq);
  } catch (error) {
    setLink("连接断开,可能已按 Ctrl+C", true);
  }
}

start();

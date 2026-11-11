# 公开版的来源

本仓库是私有仓库 `research-assistant-agent-private` 的导出副本:每个提交都是私有仓库某个提交的一次完整导出,本文件所在的提交对应下文「导出」一节所列的私有提交。计划、预登记与 README 中引用的提交号(如 `0054def`)、`docs/m*-进度.md` 等进度记录、测试集与访问日志都在私有仓库中;评审者可以申请查看私有仓库,用下面的提交号与 blob 核对预登记先于结果。时间一律为 UTC。

## 导出

- 来源:私有仓库提交 `eb0a418b75353e23d8dace601c5134dab54fe6ea`(提交时间 2026-11-11T19:04:33Z)。
- 复制的范围:`src/`、`scripts/`、`tests/`、`configs/`、`.github/`、`third_party_patches/`、项目配置、`LICENSE`、README、`docs/plan.md` 与五份里程碑计划、六份预登记、路由曲线图、`data/corpus/` 下的主题表与语料清单(`topics.json`、`manifest.json`)、`data/public/` 下的许可说明与数据来源清单、`results/` 下的汇总结果与各次打开测试集所用的配置。
- 不复制:`data/` 下的评测集、测试集及其访问日志与完成标记、开发集(含 arXiv 正文节选与由公开数据集改写的记录)、录像、术语表数据、合成数据与私有排除表;`runs/`、`models/`、`notes/`、`tools/`;`docs/` 下的进度记录、模块说明、选题记录与草案;`results/` 下的逐题结果、测试结果与含本机绝对路径的运行记录。这些规则在导出时由 git 的排除路径执行。
- 文本改动:见文末「改动的类别」。每份改动过的计划与预登记末尾都有一段公开版编辑说明。
- 计划与预登记由 Claude Code 起草,文中的「我」指 Claude Code,「他」「你」指所有者(见 README「分工」)。

## 测试集的打开

README 与下表的次数由私有仓库的 `scripts/check_readme_numbers.py` 在导出时对访问日志核对过:v1 单步测试集 <!-- num:test_set.opens -->3<!-- /num --> 次,v2 单步测试集 <!-- num:test_set_v2.opens -->1<!-- /num --> 次(端到端测试集与单步测试集在同一会话中打开)。不计入打开次数、也不运行任何系统的读取(M3 之前的结构字段读取与契约测试的读取,M7 对已用完三次打开的 v1 测试集所做的退役读取查重与重建核对)记在 `docs/m3-预登记.md` 与 `docs/m7-预登记.md` 的「披露」一节。

| 测试集族 | 第几次 | 里程碑 | 记录这次打开的私有提交 | 提交时间(UTC) |
|---|---|---|---|---|
| v1 | 1 | M3 | `0054def5165ccc7a0311bb047d2cea0f2bba1974` | 2026-10-17T04:36:22Z |
| v1 | 2 | M5 | `89e98307b2e7efb1e21943fd06174e8e62c3fdfe` | 2026-10-17T11:41:20Z |
| v1 | 3 | M6 | `2ea21b2d032c7a389dbd48561206e3581f5496c7` | 2026-10-18T18:26:27Z |
| v2 | 1 | M7 | `cb68e5adbc4677c1847c0a468c0b99a39abb88b2` | 2026-11-02T10:29:41Z |

打开所用的配置在本仓库的 `results/` 下;运行器把提交号与配置文件的 SHA-256 写进访问日志的目的栏:

| 配置 | SHA-256 |
|---|---|
| `results/test-config-m3.json` | `fa0ed67b7b9f31d283cde02e0caa2181c1d6220c7db590adc676755b5a8561e8` |
| `results/test-config-m5.json` | `3623141a320fbc20d6ffacf1f54e43844b129aafa6b945b386a5d01b8185260b` |
| `results/test-config-m6.json` | `31d644ef684f0e24d8f7039a9b1c70f60cd8ac0f9972f5e0b47a1d7a1e170277` |
| `results/test-config-m7.json` | `eea2c23f7f0b4f4f8c95246b0c980de0dc6a532630515e6482aa73eb5c4281ec` |

## 锁

各评测集按文件字节的 SHA-256 加锁,集合本身只在私有仓库中。

| 集合 | SHA-256 | 依据 |
|---|---|---|
| v1 单步测试集 | `18914bfbd81888f4…` | `docs/m3-预登记.md` 原文(原文只记前几位) |
| v1 端到端测试集 | `99e257d0…` | `docs/m3-预登记.md` 原文(原文只记前几位) |
| v2 单步测试集 | `7b94012b66ad4e460e72a5ca9e22880bbd87078d9c2b1ea7f52dab96c119aa24` | `docs/m7-预登记.md` 原文 |
| v2 端到端测试集 | `7c331570a01145136f07828c641c728847ad77b8c8f8f9d82e2b324842a8ebec` | `docs/m7-预登记.md` 原文 |
| dev-v2(开发) | `a61cff68…` | `docs/m7-预登记.md` 原文(原文只记前几位) |
| dev-v2-ext(留出) | `cb8be1af…` | `docs/m7-预登记.md` 原文(原文只记前几位) |
| 开发集 B(M8 选点) | `9a7e912bbce9008026f64d902a95cdef6f13b77842d21931c5d202e10e7cf1f5` | `docs/m8-预登记.md` 原文 |
| dev(开发) | `030586e55fb230e0b4d68038c5580d311fbeb80dad6ce474a9b57cb0db524763` | 私有仓库中的锁文件 |
| dev-ext(留出) | `89463e10054a7c9a3e121392c19379da342b976c32a0e5a11e819989677001d5` | 私有仓库中的锁文件 |
| e2e-dev(开发端到端) | `f9aea1731603333d44be037991099f22311472167c5544e5b38313440cabf96e` | 私有仓库中的锁文件 |
| 开发集 A(M8 数据修订) | `ff167c2f9a8c0ace44c8e6bb1c4f962b2a9d5d25c65cd7ae255a02151de433e3` | 私有仓库中的锁文件 |

## 计划的批准

M3、M4、M5 没有单独的计划文件:它们的计划与批准写在私有的进度记录(`docs/m3-进度.md` 至 `docs/m5-进度.md`)中,M3、M5 预登记所说的计划即指这些记录;两份预登记的提交见下文「预登记的提交」。

| 计划 | 批准(文中所记) | 改动过本文件的私有提交(UTC) |
|---|---|---|
| `docs/plan.md` | 2026-10-02(所有者的选择,「已定」一节) | `5752a8ca2241` 2026-10-03T19:18:35Z<br>`22b18da597c7` 2026-10-13T15:44:51Z |
| `docs/m6-计划.md` | 2026-10-17 | `646395f4491e` 2026-10-17T18:30:01Z<br>`d461b3774a9a` 2026-10-17T19:49:34Z |
| `docs/m7-计划.md` | 2026-10-19(修订版;原版同日先前批准,见 `e13a81a` 中的本文件) | `e13a81a6019b` 2026-10-19T07:00:54Z<br>`764f7480e51a` 2026-10-19T12:15:37Z |
| `docs/m7-本地提升计划.md` | 2026-10-30 | `3da626502825` 2026-10-30T04:34:12Z |
| `docs/m8-计划.md` | 2026-11-02 | `f0233177f40b` 2026-11-03T07:33:52Z |
| `docs/m9-计划.md` | 2026-11-10 | `46c67cffc275` 2026-11-10T05:34:32Z |

## 预登记的提交

下表列出改动过每份预登记的全部私有提交与该提交中这份文件的 git blob;打开测试集或得出结果的提交见上文。

| 预登记 | 私有提交 | 提交时间(UTC) | 本文件的 blob |
|---|---|---|---|
| `docs/m3-预登记.md` | `be8968bf80306bc2f550d92935753f6e0652368d` | 2026-10-16T20:52:07Z | `7c70369cdf854b3c37eb8f8fd1faac7b5f17fe13` |
| `docs/m3-预登记.md` | `32e0e7f214db5a0547f5da0092e977a2e5366284` | 2026-10-16T21:44:02Z | `0ac849e530e118910cab2a538c182c634ed9da3f` |
| `docs/m5-预登记.md` | `c06860e32422af05f6b2fc2f26f0b9a08915de18` | 2026-10-17T09:31:35Z | `67283c841108bcd6479610815244f154403be5c0` |
| `docs/m5-预登记.md` | `bd281b0751620b9ef80510854f186343607b6b4f` | 2026-10-17T09:38:56Z | `7e2d9f99b39ae0c3b0ca9220afcdcc40c1c6b911` |
| `docs/m5-预登记.md` | `cfb4b33a44aab03290b145e4d77cfbad5a4697d6` | 2026-10-17T09:46:09Z | `3562bd7b3dc97a42d995ece196aa4382023c46d5` |
| `docs/m5-预登记.md` | `8d252b2fbc7e59ab336ca79cbd30d61a2abfaa9f` | 2026-10-17T10:58:37Z | `8600480ae8ee2060eeabf4c4b1d78ec104a0ccfd` |
| `docs/m6-预登记.md` | `c3b684f298494040f34dfcd22fd4a94282a3f198` | 2026-10-18T16:03:10Z | `70f6cda7bd110a3c98ae5bb604dfbe26caa07273` |
| `docs/m6-预登记.md` | `864389cbf24d8ba8b89544c06d47ffbbc33a39ef` | 2026-10-18T16:37:26Z | `a103f8612164c0bcd5d9aa9bb44a4da381c4b64c` |
| `docs/m6-预登记.md` | `c2ee1e9ebb67fd414ba1da49293414cc15ec7ffc` | 2026-10-18T17:11:55Z | `a77a2ba3e3ce80a275e0377a4597222741cf835c` |
| `docs/m7-预登记.md` | `608a069b512f6f01e73649b30642c3178c5e9e5e` | 2026-10-27T06:18:36Z | `f90a4ec1fc50a56d0adcea997597c65e0fc2fe5e` |
| `docs/m7-预登记.md` | `2dfe009264d2608e778650cb251d4b0e28b2ef28` | 2026-10-29T17:59:18Z | `6cbe57eea0be2c87bd7788db458933aadc621c3b` |
| `docs/m7-预登记.md` | `4089b5c86b6f6c3252b22a460e9897691c860c93` | 2026-10-30T16:12:39Z | `55468b313ca4b2dc95532f03eceea3fa5f8c3311` |
| `docs/m7-预登记.md` | `fb9a1b9c0b3f07c922ea84e7163ba93bf02de8db` | 2026-10-30T16:22:53Z | `f735bbdbf5dddd43389e818f20d8f104da541fed` |
| `docs/m7-预登记.md` | `be7de6dd26ee99d4678de5d090b71af9be71c956` | 2026-11-01T03:40:24Z | `76c835301a015d57fe54b0c7dc9bb127d68d445d` |
| `docs/m7-底座对比预登记.md` | `764f7480e51a2b18a36ebdac5fe69a458d41ce07` | 2026-10-19T12:15:37Z | `10ca3720047414cb38913ca84a1ad3bedb811742` |
| `docs/m7-底座对比预登记.md` | `677bc0fb2714ccdb6ed474d82daec0a242d62e72` | 2026-10-26T18:42:02Z | `e4ca1054356b69b22fa3c4ff85396253871121e9` |
| `docs/m7-底座对比预登记.md` | `33963f9c33929c353983fb01767f6cf9e68cacbb` | 2026-10-26T19:02:04Z | `c0b906de7958071cbfcb288ae6d0e7f1bcbed3a5` |
| `docs/m7-底座对比预登记.md` | `a4dac8018c5b49954f8871ae19a41e0d77b08851` | 2026-10-27T05:32:46Z | `5911395f88ddd6237775fe5c8eff76143fe0a569` |
| `docs/m8-预登记.md` | `f0233177f40bbc547c741e3f531b48f906c36678` | 2026-11-03T07:33:52Z | `6bcf9ef38f0e459ce4e3b74023eed4e48089c829` |
| `docs/m8-预登记.md` | `0ae159fab6844636c706ff3926f3cbef85c7a0b3` | 2026-11-04T11:18:55Z | `a01b4ff847d97159349ef152c911038376620a88` |
| `docs/m8-预登记.md` | `e5788dac680699ba3a35af39e77fdd35a2c2205f` | 2026-11-04T11:28:12Z | `1c2ff8062331c39928bda97f0768e5fff315ecb3` |

## 改动的类别

公开版对私有仓库提交 `eb0a418` 中的文件做的文字改动,按文件与类别计数(每处改动是一条逐字匹配、恰好命中一次的替换规则,一条规则涉及几个类别就在每个类别下各计一次;规则本身留在私有的导出脚本中):

| 文件 | 类别 | 处数 |
|---|---|---|
| `README.md` | 公开版说明 | 9 |
| `README.md` | 指向私有文档的链接 | 24 |
| `README.md` | 编程助手的工具细节 | 1 |
| `README.md` | 账户与计费 | 1 |
| `data/corpus/manifest.json` | 个人细节 | 20 |
| `data/corpus/topics.json` | 个人细节 | 20 |
| `docs/m6-计划.md` | 个人细节 | 1 |
| `docs/m6-计划.md` | 删去时刻与时段 | 6 |
| `docs/m6-计划.md` | 本机细节 | 1 |
| `docs/m6-计划.md` | 本机路径 | 1 |
| `docs/m6-计划.md` | 编程助手的工具细节 | 2 |
| `docs/m6-计划.md` | 账户与计费 | 1 |
| `docs/m6-预登记.md` | 时刻改写为 UTC | 1 |
| `docs/m6-预登记.md` | 编程助手的工具细节 | 1 |
| `docs/m6-预登记.md` | 账户与计费 | 1 |
| `docs/m7-底座对比预登记.md` | 删去时刻与时段 | 1 |
| `docs/m7-底座对比预登记.md` | 时区标注 | 1 |
| `docs/m7-底座对比预登记.md` | 账户与计费 | 1 |
| `docs/m7-本地提升计划.md` | 内部编号 | 1 |
| `docs/m7-本地提升计划.md` | 删去时刻与时段 | 4 |
| `docs/m7-本地提升计划.md` | 时区标注 | 1 |
| `docs/m7-本地提升计划.md` | 本机细节 | 2 |
| `docs/m7-本地提升计划.md` | 编程助手的工具细节 | 4 |
| `docs/m7-计划.md` | 个人细节 | 1 |
| `docs/m7-计划.md` | 删去时刻与时段 | 5 |
| `docs/m7-计划.md` | 时区标注 | 3 |
| `docs/m7-计划.md` | 账户与计费 | 1 |
| `docs/m7-预登记.md` | 个人细节 | 1 |
| `docs/m7-预登记.md` | 内部编号 | 1 |
| `docs/m7-预登记.md` | 删去时刻与时段 | 9 |
| `docs/m7-预登记.md` | 时刻改写为 UTC | 9 |
| `docs/m7-预登记.md` | 时区标注 | 3 |
| `docs/m7-预登记.md` | 本机细节 | 4 |
| `docs/m7-预登记.md` | 编程助手的工具细节 | 1 |
| `docs/m7-预登记.md` | 非决定的原话 | 2 |
| `docs/m8-计划.md` | 时区标注 | 1 |
| `docs/m8-计划.md` | 编程助手的工具细节 | 1 |
| `docs/m8-预登记.md` | 内部编号 | 1 |
| `docs/m8-预登记.md` | 时刻改写为 UTC | 7 |
| `docs/m8-预登记.md` | 时区标注 | 3 |
| `docs/m8-预登记.md` | 本机细节 | 2 |
| `docs/m8-预登记.md` | 本机路径 | 1 |
| `docs/m8-预登记.md` | 私有排除表 | 8 |
| `docs/m8-预登记.md` | 编程助手的工具细节 | 3 |
| `docs/plan.md` | 个人细节 | 16 |
| `docs/plan.md` | 个人背景 | 6 |
| `docs/plan.md` | 删去时刻与时段 | 1 |
| `docs/plan.md` | 本机细节 | 2 |
| `docs/plan.md` | 本机路径 | 2 |
| `docs/plan.md` | 编程助手的工具细节 | 2 |
| `docs/plan.md` | 非决定的原话 | 1 |
| `scripts/finalize_eval_sets.py` | 内部编号 | 1 |
| `src/research_agent/agent/rules.py` | 本机路径 | 1 |
| `src/research_agent/agent/vote.py` | 本机路径 | 1 |
| `src/research_agent/data/corpus.py` | 个人细节 | 1 |
| `src/research_agent/notes.py` | 个人细节 | 1 |
| `tests/conftest.py` | 公开版说明 | 2 |
| `tests/data/test_corpus.py` | 个人细节 | 6 |
| `tests/llm/test_budget.py` | 个人细节 | 1 |
| `tests/llm/test_budget.py` | 时区标注 | 1 |
| `tests/test_notes.py` | 个人细节 | 1 |

另由导出加入的文件:本文件、`scripts/check_public_numbers.py` 与它的测试 `tests/eval/test_check_public_numbers.py`;`.pre-commit-config.yaml` 与 `.github/workflows/ci.yml` 换成公开版的数字核对。

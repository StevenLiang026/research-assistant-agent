"""Launch llama-server with the Qwen3 GGUF and an optional LoRA adapter, and wait until it is ready.

Reproduces the S1 and S1b server command (docs/m0-验证.md): -ngl 99, -c 8192, --jinja, bound to 127.0.0.1.
With --adapter the adapter is loaded and active by default: llama.cpp b11349 applies a loaded adapter at
scale 1.0 to every request without a "lora" field, even when the server was started with
--lora-init-without-apply (measured in S1b; the flag is still passed, as in that measurement). The base model
is reached only by an explicit scale 0, which LocalClient sends through lora_scale (it refuses a request that
gives no scale while an adapter is loaded). A harness that cannot send the field, such as BFCL, compares
against the base model through a second server started without --adapter. Readiness is the server's own log
line "listening on http", which b11349 prints once the model is loaded; it is read from the merged stdout and
stderr with a timeout. llama-server also keeps a cache of past prompts in system memory (--cache-ram,
8,192 MiB by default in b11349); on this 16 GB machine the BFCL baseline run of 2026-10-10 filled it within
three minutes, ran system memory down to nothing and ended after 29 minutes in a CUDA out-of-memory crash, so
the cache is off unless --cache-ram asks for it (a slot still reuses the prefix it shares with its previous
prompt, --cache-prompt). This script sends no request itself: that is the client's business
(research_agent.llm.local.LocalClient). It prints "ready" and keeps the server running until Ctrl+C; --once
stops the server right after the readiness check, a smoke test of the model and adapter files. Arguments
after "--" go to llama-server as they are, e.g. --chat-template-kwargs for ACEBench. The llama-server binary
lives outside the repository: --llama-server or the LLAMA_SERVER environment variable names it, and the
script stops with exit code 2 when neither is set. The command line, the start, the readiness check and the
stop are research_agent.llm.server's, which scripts/select_checkpoint.py shares; this script is its command
line for a server kept running by hand.

Run: .venv/Scripts/python.exe scripts/serve_local.py [--adapter runs/train/main/adapter-f16.gguf] [--once]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from research_agent.llm.server import (
    MODEL,
    SERVER_VAR,
    llama_server,
    pump,
    server_command,
    start,
    stop,
    wait_ready,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Launch llama-server locally and wait until it is ready.")
    ap.add_argument("--model", default=MODEL, help="base model GGUF")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter GGUF; loaded and active: a request without a lora scale uses it, the "
                         "base model needs scale 0 (LocalClient) or a server started without --adapter")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("-c", "--ctx-size", dest="ctx_size", type=int, default=8192,
                    help="server context size, shared by its slots")
    ap.add_argument("--ngl", type=int, default=99, help="layers kept in VRAM")
    ap.add_argument("--cache-ram", type=int, default=0,
                    help="MiB of system memory for llama-server's cache of past prompts; 0 (default here) "
                         "turns it off, llama-server's own default is 8192")
    ap.add_argument("--llama-server", default=None,
                    help=f"llama-server executable (default: ${SERVER_VAR}; one of the two is required)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for readiness")
    ap.add_argument("--once", action="store_true", help="stop the server right after it is ready")
    ap.add_argument("server_args", nargs="*", help="extra llama-server arguments, after --")
    return ap


def build_command(args: argparse.Namespace, env: Mapping[str, str] = os.environ) -> list[str]:
    return server_command(llama_server(args.llama_server, env), model=args.model, adapter=args.adapter,
                          port=args.port, ctx_size=args.ctx_size, ngl=args.ngl, cache_ram=args.cache_ram,
                          extra=args.server_args)


def main(argv: list[str] | None = None, *, popen=subprocess.Popen) -> int:
    args = build_parser().parse_args(argv)
    try:
        cmd = build_command(args)
    except LookupError as exc:
        print(exc)
        return 2
    if not Path(cmd[0]).is_file():
        print(f"llama-server not found: {cmd[0]} (--llama-server or {SERVER_VAR})")
        return 2
    print("starting:", subprocess.list2cmdline(cmd), flush=True)
    proc = start(cmd, popen=popen)
    try:
        ready, lines = wait_ready(proc, timeout=args.timeout)
        if not ready:
            if proc.poll() is None:
                print(f"llama-server did not become ready within {args.timeout:g} s")
            else:
                print(f"llama-server exited with {proc.returncode} before becoming ready")
            stop(proc)
            return 1
        print("ready", flush=True)
        if args.once:
            stop(proc)
            return 0
        return pump(proc, lines)
    except KeyboardInterrupt:
        stop(proc)
        return 0


if __name__ == "__main__":
    sys.exit(main())

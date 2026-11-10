"""scripts/export_adapter.py offline (M7 WP-E): the base model taken from the training run's manifest, and
--dry-run, through the converter when it has the option or as a local check of the adapter and the base
against the converter's registry. Nothing here runs llama.cpp's converter: run is a stand-in that records the
command."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export = load_script("export_adapter")

MODEL_4B = "models/Qwen3-4B-Thinking-2507"
ENV = {"LLAMA_CPP_SRC": "src"}


def adapter_folder(path: Path, *, config: dict | None = None) -> Path:
    """A PEFT adapter folder as scripts/train.py saves one; config replaces adapter_config.json's content."""
    path.mkdir(parents=True, exist_ok=True)
    content = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}
    (path / "adapter_config.json").write_text(json.dumps(config if config is not None else content),
                                              encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def run_folder(root: Path, model: str | None = MODEL_4B) -> Path:
    """A training run runs/train/x with its manifest (config.model, when given), the final adapter and a
    trainer checkpoint."""
    run = root / "runs" / "train" / "x"
    run.mkdir(parents=True)
    config = {} if model is None else {"model": model}
    (run / "manifest.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    adapter_folder(run / "adapter")
    adapter_folder(run / "trainer" / "checkpoint-274")
    return run


def source_tree(root: Path, *, dry_run: bool, registered: str = '"Qwen3ForCausalLM", "Qwen3Model"') -> Path:
    """A llama.cpp source tree stand-in: the converter (with or without the --dry-run option in its text)
    and a conversion package whose one module registers architectures the way the real ones do."""
    src = root / "src"
    (src / "conversion").mkdir(parents=True)
    option = '    parser.add_argument("--dry-run", action="store_true")\n' if dry_run else ""
    (src / "convert_lora_to_gguf.py").write_text("# fake converter\n" + option, encoding="utf-8")
    (src / "convert_hf_to_gguf.py").write_text('@ModelBase.register("LlamaForCausalLM")\nclass X: pass\n',
                                               encoding="utf-8")
    (src / "conversion" / "qwen.py").write_text(f"@ModelBase.register(\n    {registered}\n)\nclass Q: pass\n"
                                                "\n\n@ModelBase.register(\"Qwen3MoeForCausalLM\")\n"
                                                "class M: pass\n", encoding="utf-8")
    return src


def base_folder(root: Path, architectures: list[str]) -> Path:
    base = root / "models" / "base"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": architectures}), encoding="utf-8")
    return base


class Recorder:
    """subprocess.run stand-in: records every command; writes the output file unless told not to."""

    def __init__(self, *, writes: bool = True, code: int = 0) -> None:
        self.commands: list[list[str]] = []
        self.writes, self.code = writes, code

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess:
        self.commands.append(list(cmd))
        if self.writes:
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"x" * 100)
        return subprocess.CompletedProcess(cmd, self.code)


def test_the_base_defaults_to_the_run_manifests_model_for_an_adapter_under_a_run_folder(tmp_path,
                                                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_folder(tmp_path)
    for adapter in ("runs/train/x/trainer/checkpoint-274", "runs/train/x/adapter"):
        args = export.build_parser().parse_args([adapter])
        assert args.base == export.MODEL  # the parser's default is unchanged
        base, origin = export.base_model(args)
        assert (base, origin) == (MODEL_4B, "runs/train/x/manifest.json"), adapter
        cmd = export.command(args, python="py", env=ENV)
        assert cmd[cmd.index("--base") + 1] == MODEL_4B
    args = export.build_parser().parse_args(["runs/train/x/adapter", "--base", "models/Other"])
    assert export.base_model(args) == ("models/Other", "--base")
    assert export.command(args, python="py", env=ENV)[3] == "models/Other"
    # an explicit --base wins even when it names the default's folder: it is no default then
    args = export.build_parser().parse_args(["runs/train/x/adapter", "--base", export.MODEL])
    assert export.base_model(args) == (export.MODEL, "--base")
    assert export.command(args, python="py", env=ENV)[3] == export.MODEL


def test_without_a_run_manifest_naming_a_model_the_base_stays_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loose = adapter_folder(tmp_path / "loose")
    assert export.base_model(export.build_parser().parse_args([str(loose)])) == (export.MODEL, "the default")
    run_folder(tmp_path, model=None)  # a manifest whose config names no model
    args = export.build_parser().parse_args(["runs/train/x/adapter"])
    assert export.base_model(args) == (export.MODEL, "the default")
    # three levels below the run: not one of its adapters
    deeper = adapter_folder(tmp_path / "runs" / "train" / "x" / "a" / "b" / "c")
    assert export.base_model(export.build_parser().parse_args([str(deeper)])) == (export.MODEL, "the default")


def test_main_names_the_base_it_converts_against_and_where_it_came_from(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_folder(tmp_path)
    src = source_tree(tmp_path, dry_run=True)
    run = Recorder()
    assert export.main(["runs/train/x/trainer/checkpoint-274", "--llama-cpp-src", str(src)], run=run) == 0
    out = capsys.readouterr().out
    assert f"base {MODEL_4B} (runs/train/x/manifest.json)" in out
    (cmd,) = run.commands
    assert cmd[cmd.index("--base") + 1] == MODEL_4B and "--dry-run" not in cmd


def test_main_creates_the_outfiles_folder_before_the_converter_writes_and_a_dry_run_does_not(tmp_path,
                                                                                           monkeypatch):
    # llama.cpp's converter opens --outfile without creating its folder: on 10-28 select_checkpoint.py
    # --checkpoints exported FT3's first checkpoint into runs/train/ft3/gguf, which no step had created yet,
    # and the converter stopped with FileNotFoundError (the Recorder writes the file the same way)
    monkeypatch.chdir(tmp_path)
    run_folder(tmp_path)
    src = source_tree(tmp_path, dry_run=True)
    folder = tmp_path / "runs" / "train" / "x" / "gguf"
    outfile = "runs/train/x/gguf/checkpoint-274-f16.gguf"
    argv = ["runs/train/x/trainer/checkpoint-274", "--llama-cpp-src", str(src), "--outfile", outfile]
    assert export.main([*argv, "--dry-run"], run=Recorder(writes=False)) == 0
    assert not folder.exists()  # a dry run writes nothing, no folder either
    run = Recorder()
    assert export.main(argv, run=run) == 0
    (cmd,) = run.commands
    assert cmd[cmd.index("--outfile") + 1] == str(Path(outfile)) and (tmp_path / outfile).is_file()


def test_dry_run_passes_through_to_a_converter_that_has_the_option_and_writes_nothing(tmp_path, monkeypatch,
                                                                                     capsys):
    monkeypatch.chdir(tmp_path)
    run_folder(tmp_path)
    src = source_tree(tmp_path, dry_run=True)
    assert export.converter_supports_dry_run(src / "convert_lora_to_gguf.py")
    run = Recorder(writes=False)
    code = export.main(["runs/train/x/adapter", "--llama-cpp-src", str(src), "--dry-run"], run=run)
    out = capsys.readouterr().out
    (cmd,) = run.commands
    assert code == 0 and cmd[-1] == "runs/train/x/adapter" and cmd[-2] == "--dry-run"
    assert "dry run" in out and not (tmp_path / "runs" / "train" / "x" / "adapter-f16.gguf").exists()
    failing = Recorder(writes=False, code=1)
    assert export.main(["runs/train/x/adapter", "--llama-cpp-src", str(src), "--dry-run"], run=failing) == 1


def test_dry_run_on_a_converter_without_the_option_checks_the_adapter_and_the_registry_locally(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    src = source_tree(tmp_path, dry_run=False)
    assert not export.converter_supports_dry_run(src / "convert_lora_to_gguf.py")
    assert export.registered_architectures(src) == {"LlamaForCausalLM", "Qwen3ForCausalLM", "Qwen3Model",
                                                    "Qwen3MoeForCausalLM"}
    adapter = adapter_folder(tmp_path / "adapter")
    base = base_folder(tmp_path, ["Qwen3ForCausalLM"])
    run = Recorder()
    argv = [str(adapter), "--llama-cpp-src", str(src), "--base", str(base), "--dry-run"]
    assert export.main(argv, run=run) == 0 and run.commands == []
    out = capsys.readouterr().out
    assert "Qwen3ForCausalLM" in out and "registered" in out and "q_proj" in out
    assert not (tmp_path / "adapter-f16.gguf").exists()


def test_the_local_check_names_an_unregistered_architecture_and_a_malformed_adapter(tmp_path, monkeypatch,
                                                                                    capsys):
    monkeypatch.chdir(tmp_path)
    src = source_tree(tmp_path, dry_run=False)
    adapter = adapter_folder(tmp_path / "adapter")
    base = base_folder(tmp_path, ["SparkForCausalLM"])
    argv = [str(adapter), "--llama-cpp-src", str(src), "--base", str(base), "--dry-run"]
    assert export.main(argv, run=Recorder()) == 1
    assert "SparkForCausalLM" in capsys.readouterr().out
    base = base_folder(tmp_path / "other", ["Qwen3ForCausalLM"])
    (adapter / "adapter_config.json").unlink()
    argv = [str(adapter), "--llama-cpp-src", str(src), "--base", str(base), "--dry-run"]
    assert export.main(argv, run=Recorder()) == 1
    assert "adapter_config.json" in capsys.readouterr().out
    adapter_folder(adapter, config={"peft_type": "ADALORA", "target_modules": ["q_proj"]})
    assert export.main(argv, run=Recorder()) == 1
    assert "ADALORA" in capsys.readouterr().out
    adapter_folder(adapter, config={"peft_type": "LORA", "target_modules": []})
    assert export.main(argv, run=Recorder()) == 1
    assert "target_modules" in capsys.readouterr().out
    (adapter / "adapter_model.safetensors").unlink()
    adapter_folder(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    assert export.main(argv, run=Recorder()) == 1
    assert "adapter_model.safetensors" in capsys.readouterr().out

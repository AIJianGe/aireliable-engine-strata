"""``strata-<tag>.json`` 配置落地（_patch_model_config）与两段式映射测试（阶段 3 / 阶段 7）。

覆盖：
    - ``_mode_to_model``：模式 → (family, model, tag, mtp, vision) 映射
    - ``_patch_model_config``：ctx/port 原子落盘；spec=0 只剔除 --mtp（保留 --spec/--spec-min-p）；
      基线机制避免多模式共用同一 json 时互相污染
    - ``_validate_source``：源码脚本缺失抛 RuntimeError
    - config 路径解析：setup_script / server_script / model_config_json / strata_python
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from engines.strata_llm_7c4f2b91.config import EngineConfig, parse_args
from engines.strata_llm_7c4f2b91.strata_process import StrataProcessManager

INTERNAL_PORT = 8123


@pytest.fixture
def cfg(tmp_path):
    # 源码目录指向 tmp（不存在），使 _validate_source 等测试保持密闭
    return EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=tmp_path / "bins",
        strata_data_dir=tmp_path / "data",
        strata_src_dir=tmp_path / "strata-src",
        internal_port=INTERNAL_PORT,
    )


@pytest.fixture
def pm(cfg):
    mgr = StrataProcessManager(cfg)
    mgr._internal_port = INTERNAL_PORT
    return mgr


# ===== 模式 → 量化档映射 =====

def test_mode_to_model_mapping():
    m = StrataProcessManager._mode_to_model
    assert m("strata/fast") == ("qwen", "Q2_0", "Q2_0", False, False)
    assert m("strata/balanced") == ("qwen", "IQ2_XS", "IQ2_XS", True, True)
    assert m("strata/quality") == ("qwen", "IQ3_XXS", "IQ3_XXS", False, False)
    assert m("strata/long") == ("qwen", "IQ2_XS", "IQ2_XS", False, False)
    assert m("strata/compact") == ("qwen", "IQ2_XS", "IQ2_XS", False, False)
    assert m("strata/coder") == ("coder", "IQ1_M", "coder-IQ1_M", False, False)


def test_mode_to_model_unknown_raises():
    with pytest.raises(RuntimeError):
        StrataProcessManager._mode_to_model("strata/does-not-exist")


# ===== 配置 json 落地（_patch_model_config）=====

def _write_cfg(path, args):
    path.write_text(json.dumps({"port": 9999, "args": args}), encoding="utf-8")
    return path


def test_patch_model_config_sets_ctx_port_and_keeps_spec(pm, tmp_path):
    cfg_json = _write_cfg(
        tmp_path / "strata-iq2_xs.json",
        ["--pack", "p", "--max-context", "131072",
         "--spec", "3", "--spec-min-p", "0.5", "--mtp", "m"],
    )
    pm._patch_model_config(cfg_json, ctx=262144, port=INTERNAL_PORT, spec=4)

    # 原子写：不应残留 .tmp
    assert not (tmp_path / "strata-iq2_xs.json.tmp").exists()

    data = json.loads(cfg_json.read_text(encoding="utf-8"))
    assert data["port"] == INTERNAL_PORT
    args = data["args"]
    assert args[args.index("--max-context") + 1] == "262144"
    assert args[args.index("--spec") + 1] == "4"
    assert "--mtp" in args          # spec>0 保留 MTP
    assert "--spec-min-p" in args


def test_patch_model_config_spec_zero_drops_mtp(pm, tmp_path):
    """spec=0（关闭 MTP 草稿头）只剔除 --mtp；--spec/--spec-min-p 必须保留。

    回归点：``--spec T (T>=2)`` 是原生 IQ pack 多 token 解码内核的硬性要求，
    误删会导致 ``strata.exe`` 以 exit code 2 拒绝启动（实测）。
    """
    cfg_json = _write_cfg(
        tmp_path / "strata-iq2_xs.json",
        ["--max-context", "131072", "--mtp", "m", "--spec", "4", "--spec-min-p", "0.5"],
    )
    pm._patch_model_config(cfg_json, ctx=262144, port=INTERNAL_PORT, spec=0)

    data = json.loads(cfg_json.read_text(encoding="utf-8"))
    assert data["port"] == INTERNAL_PORT
    args = data["args"]
    assert args[args.index("--max-context") + 1] == "262144"
    assert "--mtp" not in args                      # 关闭 MTP 草稿头
    assert args[args.index("--spec") + 1] == "4"     # 保留 --spec
    assert "--spec-min-p" in args                   # 保留 --spec-min-p


def test_patch_model_config_baseline_prevents_cross_mode_pollution(pm, tmp_path):
    """多模式共用同一 json：后续 patch 从 `.baseline` 派生，不继承前一次改动。

    回归点：long（spec=0）删掉 --mtp 后，balanced（spec=4）必须能拿回 --mtp（实测）。
    """
    cfg_json = _write_cfg(
        tmp_path / "strata-iq2_xs.json",
        ["--max-context", "131072", "--mtp", "m", "--spec", "4", "--spec-min-p", "0.5"],
    )
    # 第 1 次：long —— 关闭 MTP
    pm._patch_model_config(cfg_json, ctx=262144, port=INTERNAL_PORT, spec=0)
    args = json.loads(cfg_json.read_text(encoding="utf-8"))["args"]
    assert "--mtp" not in args

    # 第 2 次：balanced —— 开启 MTP，须从基线恢复 --mtp
    pm._patch_model_config(cfg_json, ctx=131072, port=INTERNAL_PORT, spec=4)
    data = json.loads(cfg_json.read_text(encoding="utf-8"))
    args = data["args"]
    assert args[args.index("--max-context") + 1] == "131072"
    assert args[args.index("--spec") + 1] == "4"
    assert "--mtp" in args
    assert (tmp_path / "strata-iq2_xs.json.baseline").exists()


def test_patch_model_config_appends_max_context_when_absent(pm, tmp_path):
    cfg_json = _write_cfg(tmp_path / "strata-iq2_xs.json", ["--pack", "p"])
    pm._patch_model_config(cfg_json, ctx=65536, port=INTERNAL_PORT, spec=0)

    args = json.loads(cfg_json.read_text(encoding="utf-8"))["args"]
    assert args[args.index("--max-context") + 1] == "65536"


# ===== 源码 / 解释器路径解析 =====

def test_config_paths(cfg):
    assert cfg.setup_script == cfg.strata_src_dir / "setup.py"
    assert cfg.server_script == cfg.strata_src_dir / "serve" / "server.py"
    assert cfg.model_config_json("IQ2_XS").name == "strata-iq2_xs.json"
    assert cfg.model_config_json("IQ2_XS").parent == cfg.strata_src_dir
    assert cfg.strata_venv_dir == cfg.strata_src_dir / ".venv"


def test_strata_python_env_override(monkeypatch, cfg):
    monkeypatch.setenv("STRATA_PYTHON", "C:/custom/python.exe")
    assert cfg.strata_python == "C:/custom/python.exe"
    # 引擎管理器同源（不触碰 venv 创建）
    assert StrataProcessManager(cfg)._ensure_strata_python() == "C:/custom/python.exe"


def test_validate_source_raises_when_missing(pm):
    with pytest.raises(RuntimeError) as ei:
        pm._validate_source()
    assert "Strata 源码脚本缺失" in str(ei.value)


# ===== 设备策略回归（平台 AIRELIABLE_DEVICE_POLICY 语义对齐，v0.3.5+）=====

def _cfg(tmp_path, device):
    return EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=tmp_path / "bins",
        strata_data_dir=tmp_path / "data",
        device=device,
    )


def _fake_torch(available):
    """构造一个假 torch 模块（仅提供 cuda.is_available）。"""
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: available)
    return mod


def _fake_smi_ok(stdout="Quadro RTX 8000"):
    """假 nvidia-smi：返回成功 + GPU 名称。"""

    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout + "\n", stderr=""
        )

    return _run


def test_resolve_device_explicit_policies(tmp_path):
    """显式策略不探测：cpu→cpu；gpu→gpu；gpu+cpu→gpu。"""
    assert _cfg(tmp_path, "cpu").resolve_device() == "cpu"
    assert _cfg(tmp_path, "gpu").resolve_device() == "gpu"
    assert _cfg(tmp_path, "gpu+cpu").resolve_device() == "gpu"


def test_resolve_device_auto_prefers_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    assert _cfg(tmp_path, "auto").resolve_device() == "gpu"


def test_resolve_device_auto_falls_back_to_nvidia_smi(tmp_path, monkeypatch):
    """本引擎常态：无 torch，必须靠 nvidia-smi 兜底，否则恒回退 cpu。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setattr(subprocess, "run", _fake_smi_ok())
    assert _cfg(tmp_path, "auto").resolve_device() == "gpu"


def test_resolve_device_auto_cpu_when_no_gpu(tmp_path, monkeypatch):
    """无 torch + 无 nvidia-smi（FileNotFoundError）→ cpu。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))

    def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _cfg(tmp_path, "auto").resolve_device() == "cpu"


def test_parse_args_device_reads_env_then_cli(tmp_path, monkeypatch):
    """设备优先级：命令行 --device > AIRELIABLE_DEVICE_POLICY > auto。"""
    base = [
        "--models-root", str(tmp_path / "models"),
        "--output-root", str(tmp_path / "output"),
        "--bins-dir", str(tmp_path / "bins"),
        "--data-dir", str(tmp_path / "data"),
    ]

    # 1) 无 env、无 CLI → auto
    monkeypatch.delenv("AIRELIABLE_DEVICE_POLICY", raising=False)
    assert parse_args(base).device == "auto"

    # 2) 有 env、无 CLI → 读 env
    monkeypatch.setenv("AIRELIABLE_DEVICE_POLICY", "gpu+cpu")
    assert parse_args(base).device == "gpu+cpu"

    # 3) CLI 覆盖 env
    assert parse_args(["--device", "cpu", *base]).device == "cpu"


def test_engine_config_normalizes_invalid_device(tmp_path):
    """非法设备策略归一化为 auto（EngineConfig.__post_init__）。"""
    assert _cfg(tmp_path, "tpu").device == "auto"

"""``_check_*_integrity`` 只校验不下载 + 抛错文案测试（阶段 4 / 阶段 7）。

覆盖：
    - bins 缺失 → RuntimeError，信息含本地路径 + 官方 Release URL
    - 模型分片缺失 → RuntimeError，信息含**缺失清单 + 本地路径 + HF 仓库 + 下载指引**
    - 编程档指向 Coder 家族仓库
    - 全程**不触发任何网络下载**（load 只校验）
"""

from __future__ import annotations

import pytest

from engines.strata_llm_7c4f2b91.config import EngineConfig
from engines.strata_llm_7c4f2b91.engine import StrataEngine


@pytest.fixture
def cfg(tmp_path):
    # 预建模型目录（使校验走到「文件不完整」分支，而非「目录不存在」）
    base = tmp_path / "models" / "llm" / "checkpoints" / "strata"
    base.mkdir(parents=True, exist_ok=True)
    return EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=tmp_path / "bins",
        strata_data_dir=tmp_path / "data",
    )


@pytest.fixture
def engine(cfg):
    eng = StrataEngine(cfg)
    assert eng.config.model_dir.exists()
    return eng


def test_bins_missing_raises_with_path_and_release_url(engine):
    with pytest.raises(RuntimeError) as ei:
        engine._check_bins_integrity()
    msg = str(ei.value)
    assert "可执行文件缺失" in msg
    assert str(engine.config.bins_dir) in msg
    assert "v0.1.45" in msg                     # 官方 Release 版本
    assert "github.com/Niko1221/Strata" in msg


def test_models_missing_raises_with_manifest_and_repo(engine):
    with pytest.raises(RuntimeError) as ei:
        engine._check_models_dir_integrity("strata/balanced")
    msg = str(ei.value)
    assert "模型文件不完整" in msg
    assert str(engine.config.model_dir) in msg
    assert "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF" in msg
    assert "下载缺失文件" in msg
    # 缺失清单应含 IQ2_XS 两个分片 + mmproj（balanced 含 vision）
    # 注：MTP draft 由 Strata setup.py 构建到数据目录，不在模型目录，故不在此清单
    assert "IQ2_XS-00001-of-00002.gguf" in msg
    assert "IQ2_XS-00002-of-00002.gguf" in msg
    assert "mmproj-Qwen3.8-Flash-Next-BF16.gguf" in msg


def test_coder_variant_points_to_coder_repo(engine):
    with pytest.raises(RuntimeError) as ei:
        engine._check_models_dir_integrity("strata/coder")
    msg = str(ei.value)
    assert "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF" in msg
    assert "IQ1_M-00001-of-00002.gguf" in msg


def test_bins_present_passes_without_network(tmp_path):
    """bins 存在时校验通过（不发起任何网络请求）。"""
    bins = tmp_path / "bins"
    bins.mkdir()
    exe = bins / ("strata.exe" if __import__("sys").platform == "win32" else "strata")
    exe.write_bytes(b"stub")

    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=bins,
        strata_data_dir=tmp_path / "data",
    )
    eng = StrataEngine(cfg)
    eng._check_bins_integrity()  # 不抛异常即通过


async def test_load_does_not_download(engine):
    """``load``（_load_impl）在缺文件时必须抛错、**不得下载**（红线 2）。"""
    with pytest.raises(RuntimeError) as ei:
        await engine._load_impl("strata/balanced")
    assert "可执行文件缺失" in str(ei.value) or "模型文件不完整" in str(ei.value)
    # 下载目录不应被创建出权重文件
    assert not any(engine.config.model_dir.glob("*.gguf"))

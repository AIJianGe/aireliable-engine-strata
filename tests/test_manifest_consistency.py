"""manifest.yaml 与 schemas.MODES 一致性测试。

覆盖任务文档阶段 5 / 阶段 7：
    - ``model_file_ids`` 并集 == ``model_files[].id`` 集合（project-rules §二.1）
    - ``variants[].name`` == ``schemas.MODES[].name``（variant 扫描匹配契约）
    - ``default_variant`` 必在 variants 中声明
"""

from __future__ import annotations

from pathlib import Path

import yaml

from engines.strata_llm_7c4f2b91.schemas import MODES

ENGINE_DIR = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    text = (ENGINE_DIR / "manifest.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_model_file_ids_union_matches_model_files():
    """variants[].model_file_ids 的并集必须等于 model_files[].id 集合。"""
    models = _manifest()["models"]
    ids = {f["id"] for f in models["model_files"]}
    union: set[str] = set()
    for v in models["variants"]:
        union |= set(v["model_file_ids"])
    assert union == ids, f"缺失={union - ids} 多余={ids - union}"


def test_variant_names_match_modes():
    """manifest variants 名必须与 schemas.MODES 逐字一致（顺序亦需一致）。"""
    models = _manifest()["models"]
    assert [v["name"] for v in models["variants"]] == [m["name"] for m in MODES]


def test_variant_model_file_ids_are_declared():
    """每个 variant 引用的 id 必须都在 model_files 中声明。"""
    models = _manifest()["models"]
    ids = {f["id"] for f in models["model_files"]}
    for v in models["variants"]:
        assert set(v["model_file_ids"]) <= ids, f"{v['name']} 引用了未声明的 id"


def test_default_variant_declared_in_variants():
    models = _manifest()["models"]
    names = [v["name"] for v in models["variants"]]
    assert models["default_variant"] in names


def test_model_files_have_source_and_mirror():
    """每个 model_file 必须声明 source.url + source.mirror_url（project-rules §二.1）。"""
    models = _manifest()["models"]
    for f in models["model_files"]:
        src = f.get("source") or {}
        assert src.get("url"), f"{f['id']} 缺 source.url"
        assert src.get("mirror_url"), f"{f['id']} 缺 source.mirror_url"
        assert src.get("repo_id"), f"{f['id']} 缺 source.repo_id"


def test_registry_declares_no_code_path():
    """红线 1：注册片段不得声明 ``code_path`` 键（否则 cwd 变引擎目录）。

    注意：注释里出现「不要声明 code_path」是允许的，此测试校验的是**解析后的
    YAML 键**不存在（而非文本子串）。
    """
    reg = yaml.safe_load((ENGINE_DIR / "registry.yaml").read_text(encoding="utf-8"))
    local = ((reg.get("deployment") or {}).get("local_config") or {})
    assert "code_path" not in local, "registry.yaml 的 local_config 不得声明 code_path"
    assert local.get("startup_cmd", "").startswith("python -m engines.")

"""SSE 帧契约 + ``_infer_impl`` 错误路径测试（阶段 4 / 阶段 7，R25）。

覆盖：
    - ``make_frame`` / ``make_final_frame`` / ``make_error_frame`` 帧形状
    - 输入缺失时 ``_infer_impl`` **真实发出 stage=error 帧**（R25：不得静默）
    - ``_infer_impl`` 内部捕获异常、**不重抛**（避免 BaseEngine.infer 置 ERROR）
"""

from __future__ import annotations

import json

import pytest

from aireliable_engine import InferRequest
from aireliable_engine.sse import (
    SSEStream,
    make_error_frame,
    make_final_frame,
    make_frame,
)

from engines.strata_llm_7c4f2b91.config import EngineConfig
from engines.strata_llm_7c4f2b91.engine import StrataEngine


@pytest.fixture
def engine(tmp_path):
    base = tmp_path / "models" / "llm" / "checkpoints" / "strata"
    base.mkdir(parents=True, exist_ok=True)
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=tmp_path / "bins",
        strata_data_dir=tmp_path / "data",
    )
    return StrataEngine(cfg)


async def _collect_frames(engine: StrataEngine, request: InferRequest) -> list[dict]:
    """执行 ``_infer_impl`` 并收集其发出的全部 SSE 帧。"""
    stream = SSEStream()
    await engine._infer_impl(request, stream)
    stream.close()
    frames: list[dict] = []
    async for raw in stream.events():
        payload = raw[len("data: "):].strip()
        frames.append(json.loads(payload))
    return frames


# ===== 帧构造函数 =====

def test_make_frame_shape():
    f = make_frame(0.5, "generating", delta="hi")
    assert f["progress"] == 0.5
    assert f["stage"] == "generating"
    assert f["delta"] == "hi"


def test_make_final_frame_shape():
    f = make_final_frame({"text": "x"}, {"total": 1.0})
    assert f["progress"] == 1.0
    assert f["stage"] == "completed"
    assert f["result"]["text"] == "x"
    assert f["timing"]["total"] == 1.0


def test_make_error_frame_shape():
    f = make_error_frame("boom", "infer_failed")
    assert f["progress"] == 0.0
    assert f["stage"] == "error"
    assert f["error"]["type"] == "infer_failed"
    assert f["error"]["message"] == "boom"


# ===== _infer_impl 错误路径（R25）=====

async def test_infer_empty_inputs_emits_error_frame(engine):
    """inputs 为空 → 单帧 stage=error（missing_input），且不重抛异常。"""
    frames = await _collect_frames(engine, InferRequest(inputs={}))
    assert len(frames) == 1
    assert frames[0]["stage"] == "error"
    assert frames[0]["error"]["type"] == "missing_input"


async def test_infer_prompt_only_without_model_emits_load_error(engine):
    """仅给 prompt 但模型未下载 → 自动加载失败，发出 stage=error（load_failed）。"""
    frames = await _collect_frames(
        engine, InferRequest(inputs={"prompt": "你好"}, variant="strata/balanced")
    )
    assert frames[-1]["stage"] == "error"
    assert frames[-1]["error"]["type"] == "load_failed"
    # 首帧应为 loading 提示
    assert frames[0]["stage"] == "loading"


async def test_infer_never_raises(engine):
    """_infer_impl 内部消化异常，永不向上抛出（否则状态机会被置 ERROR）。"""
    # 无异常即为通过
    await _collect_frames(engine, InferRequest(inputs={}))

"""Agent 工具调用透传 + 显存分层推荐 单测（任务 63 / 62-P2）。

覆盖：
    - ``_accumulate_tool_calls`` 的按 index 合并语义（id/name 覆盖式、arguments 拼接式）
    - ``ChatInputs`` 透传 ``tools`` / ``tool_choice``（extra="allow" 兼容）
    - ``_infer_impl`` 端到端：请求体带 tools → native SSE ``delta.tool_calls``
      → 进度帧透传 + 完成帧 result.tool_calls 聚合
    - ``_recommend`` 显存分层（12/8/6GB…）+ 低显存 caveat
"""

from __future__ import annotations

import json

import pytest

from aireliable_engine import InferRequest
from aireliable_engine.sse import SSEStream

from engines.strata_llm_7c4f2b91.config import EngineConfig
from engines.strata_llm_7c4f2b91.engine import StrataEngine, _accumulate_tool_calls
from engines.strata_llm_7c4f2b91.model_scanner import ModelScanner
from engines.strata_llm_7c4f2b91.schemas import ChatInputs, MODES, StrataVariantInfo


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


# ===== _accumulate_tool_calls =====

def test_accumulate_single_call_split_arguments():
    acc: dict = {}
    _accumulate_tool_calls(acc, [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": ""}},
    ])
    _accumulate_tool_calls(acc, [
        {"index": 0, "function": {"arguments": '{"city":'}},
    ])
    _accumulate_tool_calls(acc, [
        {"index": 0, "function": {"arguments": '"Paris"}'}},
    ])
    assert acc[0]["id"] == "call_1"
    assert acc[0]["type"] == "function"
    assert acc[0]["function"]["name"] == "get_weather"
    assert acc[0]["function"]["arguments"] == '{"city":"Paris"}'


def test_accumulate_parallel_calls_by_index():
    acc: dict = {}
    _accumulate_tool_calls(acc, [
        {"index": 0, "id": "a", "function": {"name": "f0", "arguments": "1"}},
        {"index": 1, "id": "b", "function": {"name": "f1", "arguments": "2"}},
    ])
    assert sorted(acc) == [0, 1]
    assert acc[1]["function"]["name"] == "f1"


def test_accumulate_ignores_non_dict_and_bad_index():
    acc: dict = {}
    _accumulate_tool_calls(acc, ["not-a-dict", {"index": "xx", "id": "z"}])
    # 非法 index 归 0；非 dict 项忽略
    assert list(acc.keys()) == [0]
    assert acc[0]["id"] == "z"


def test_accumulate_empty_is_noop():
    acc: dict = {}
    _accumulate_tool_calls(acc, [])
    _accumulate_tool_calls(acc, None)
    assert acc == {}


# ===== ChatInputs 透传字段 =====

def test_chatinputs_accepts_tools_and_tool_choice():
    inp = ChatInputs(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="auto",
    )
    assert inp.tools and inp.tools[0]["function"]["name"] == "f"
    assert inp.tool_choice == "auto"


def test_chatinputs_defaults_tools_none():
    inp = ChatInputs(messages=[{"role": "user", "content": "hi"}])
    assert inp.tools is None
    assert inp.tool_choice is None


# ===== _infer_impl 端到端（stub httpx）=====

class _FakeProc:
    pid = 4242
    def poll(self):
        return None


class _FakeResp:
    status_code = 200
    def __init__(self, lines):
        self._lines = lines
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def aiter_lines(self):
        for line in self._lines:
            yield line
    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp
    async def __aenter__(self):
        return self._resp
    async def __aexit__(self, *a):
        return False


SAMPLE_TOOL_CALL_SSE = [
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
    '"type":"function","function":{"name":"get_weather","arguments":""}}]}}]}',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"{\\"city\\":"}}]}}]}',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"Paris\\"}"}}]}}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
    'data: [DONE]',
]


def _install_fake_httpx(monkeypatch, lines):
    """替换 httpx.AsyncClient，记录最后一次请求体，回放给定 SSE 行。"""
    captured: dict = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def stream(self, method, url, json=None):
            captured["body"] = json
            return _FakeStreamCtx(_FakeResp(lines))

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return captured


def _wire_running(engine, port=9999, mode="strata/balanced"):
    """让引擎看起来已就绪（跳过自动加载），并暴露内部端点。"""
    engine.process._process = _FakeProc()
    engine.process._internal_port = port
    engine.process.current_mode = mode


async def _collect_frames(engine, request) -> list[dict]:
    stream = SSEStream()
    await engine._infer_impl(request, stream)
    stream.close()
    frames: list[dict] = []
    async for raw in stream.events():
        frames.append(json.loads(raw[len("data: "):].strip()))
    return frames


async def test_infer_forwards_tools_and_aggregates_tool_calls(engine, monkeypatch):
    captured = _install_fake_httpx(monkeypatch, SAMPLE_TOOL_CALL_SSE)
    _wire_running(engine)
    req = InferRequest(
        inputs={
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
        },
        variant="strata/balanced",
    )
    frames = await _collect_frames(engine, req)

    # 请求体透传
    assert captured["body"]["tools"] == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert captured["body"]["tool_choice"] == "auto"

    # 进度帧含顶层 tool_calls 增量（gateway 原样转发）
    gen = [f for f in frames if f.get("stage") == "generating" and f.get("tool_calls")]
    assert gen, "应至少有一帧带 tool_calls 增量"

    # 完成帧聚合完整调用
    final = frames[-1]
    assert final["stage"] == "completed"
    assert final["result"]["finish_reason"] == "tool_calls"
    tcs = final["result"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "get_weather"
    assert tcs[0]["function"]["arguments"] == '{"city":"Paris"}'
    assert tcs[0]["id"] == "call_1"


async def test_infer_without_tools_omits_key(engine, monkeypatch):
    captured = _install_fake_httpx(monkeypatch, [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ])
    _wire_running(engine)
    req = InferRequest(
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        variant="strata/balanced",
    )
    frames = await _collect_frames(engine, req)
    assert "tools" not in captured["body"]
    assert "tool_choice" not in captured["body"]
    final = frames[-1]
    assert final["stage"] == "completed"
    assert "tool_calls" not in final["result"]


async def test_infer_reasoning_goes_to_extra(engine, monkeypatch):
    """思维链走 extra.reasoning_content（gateway 流式读取契约）。"""
    _install_fake_httpx(monkeypatch, [
        'data: {"choices":[{"delta":{"reasoning_content":"think..."}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ])
    _wire_running(engine)
    req = InferRequest(
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        variant="strata/balanced",
    )
    frames = await _collect_frames(engine, req)
    gen = [f for f in frames if f.get("stage") == "generating" and f.get("extra")]
    assert gen and gen[0]["extra"]["reasoning_content"] == "think..."
    assert frames[-1]["result"]["reasoning_content"] == "think..."


# ===== _recommend 显存分层 =====

@pytest.mark.parametrize("ram,vram,expected", [
    (64, 24, "strata/quality"),
    (64, 16, "strata/quality"),   # vram≥16 且 ram≥60 → quality
    (64, 15, "strata/balanced"),  # vram 15 < 16 → balanced
    (48, 12, "strata/balanced"),
    (64, 12, "strata/balanced"),
    (32, 12, "strata/coder"),
    (32, 8, "strata/coder"),
    (64, 6, "strata/fast"),      # RAM 够、显存 <8
    (24, 6, "strata/compact"),   # RAM 也不够
])
def test_recommend_layering(engine, ram, vram, expected):
    engine._hardware = {"probe_ok": True, "ram_total_gb": ram, "vram_total_gb": vram}
    assert engine._recommend() == expected


def test_recommend_probe_failed_falls_back_balanced(engine):
    engine._hardware = {"probe_ok": False}
    assert engine._recommend() == "strata/balanced"
    assert engine._recommend_note is None


@pytest.mark.parametrize("vram,has_note", [
    (24, False),
    (12, False),
    (10, True),
    (6, True),
])
def test_recommend_low_vram_note(engine, vram, has_note):
    engine._hardware = {"probe_ok": True, "ram_total_gb": 64, "vram_total_gb": vram}
    engine._recommend()
    assert bool(engine._recommend_note) is has_note


# ===== 低显存 ⚠️ 字段（任务 62 · T2.4 / T3.2）=====

def test_low_vram_note_only_declared_for_low_tier():
    """仅「低显存档」（coder 8GB）在 MODES 中声明 low_vram_note，其余为 None。"""
    coder = next(m for m in MODES if m["name"] == "strata/coder")
    assert coder.get("low_vram_note"), "coder 档应声明低显存提示"
    assert all(
        m.get("low_vram_note") is None
        for m in MODES if m["name"] != "strata/coder"
    )


def test_strata_variant_info_low_vram_note_roundtrip():
    """StrataVariantInfo 扩展字段 low_vram_note 序列化保留（/capabilities 契约）。"""
    v = StrataVariantInfo(
        name="strata/coder", base_model="Qwen3.8-Flash-Next-Coder",
        precision="IQ1_M", loader="strata",
        vram_required="8GB", ram_required_gb=32,
        low_vram_note="⚠️ 低显存档",
    )
    assert v.model_dump()["low_vram_note"] == "⚠️ 低显存档"
    v2 = StrataVariantInfo(
        name="strata/fast", base_model="Qwen3.8-Flash-Next",
        precision="Q2_0", loader="strata", vram_required="12GB",
    )
    assert v2.model_dump()["low_vram_note"] is None


def test_model_scanner_passes_low_vram_note():
    """ModelScanner._build_variant_info 原样透传 MODES 的 low_vram_note。"""
    coder_def = next(m for m in MODES if m["name"] == "strata/coder")
    info = ModelScanner._build_variant_info(None, coder_def, True)
    assert info.low_vram_note == coder_def["low_vram_note"]
    fast_def = next(m for m in MODES if m["name"] == "strata/fast")
    assert ModelScanner._build_variant_info(None, fast_def, True).low_vram_note is None

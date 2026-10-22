"""Strata LLM 引擎 Pydantic 数据模型 + 6 模式矩阵。

扩展 SDK 的 ``VariantInfo`` / ``InferRequest.inputs``，定义 chats 引擎专属字段。

variant = 命名模式（6 个，每个模式固定「量化档 + 上下文长度 + 是否含 MTP/vision」）：

    | variant             | 模式 | 量化     | ctx     | MTP | vision | RAM  | 磁盘  |
    |---------------------|------|----------|---------|-----|--------|------|-------|
    | strata/fast         | 极速 | Q2_0     |  32768  |  ✗  |   ✗    | 48GB | 66.4G |
    | strata/balanced  ★  | 均衡 | IQ2_XS   | 131072  |  ✓  |   ✓    | 48GB | 68.0G |
    | strata/quality      | 质量 | IQ3_XXS  | 131072  |  ✗  |   ✗    | 60GB | 75.8G |
    | strata/long         | 长文 | IQ2_XS   | 262144  |  ✗  |   ✗    | 48GB | 68.0G |
    | strata/coder        | 编程 | IQ1_M    |  32768  |  ✗  |   ✗    | 32GB | 58.4G |
    | strata/compact      | 精简 | IQ2_XS   |  32768  |  ✗  |   ✗    | 48GB | 68.0G |

    均衡 / 长文 / 精简 共用 IQ2_XS 权重，仅上下文长度不同；
    编程档使用 Coder 家族权重（独立仓库）。

任务类型（task_type）：
    - chat：多轮对话生成（OpenAI 兼容 /v1/chat/completions 语义）

infer 入参（放入 ``InferRequest.inputs``，由网关 / WebUI 透传）：
    - messages     ：``[{"role": "user", "content": "..."}]`` —— **主入口**
    - prompt       ：纯文本提示（单轮兜底，会被包装成一条 user 消息）
    - max_tokens / temperature / top_p / top_k / min_p / seed / stop
    - stream       ：是否流式（默认 True；SSE 契约要求）
    - reasoning_effort ：思考强度提示（low / medium / high，透传给后端）
    - context      ：覆盖该次请求的上下文长度（一般无需传，由模式决定）
    - vision       ：是否启用多模态（需模式自带 mmproj）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo

# ===== 能力声明 =====
CAPABILITY = "chat"
ENGINE_TYPE = "llm"

# ===== 任务类型（必须与 gateway 的 task_type 逐字一致）=====
TASK_TYPE = "chat"

# ===== MTP 投机解码取值范围 =====
REASONING_EFFORTS = ("low", "medium", "high")

# ===== 默认模式 =====
DEFAULT_MODE = "strata/balanced"

# ===== 6 模式矩阵（单一真相源）=====
# ⚠️ "name" 必须与 manifest.yaml 的 models.variants[].name 逐字一致
#    （variant 扫描匹配契约，双处不一致 → UI 扫不到 → 无法触发下载）
# quant_key  : 对应 Strata setup.py MODELS 的量化档键名
# file_tag   : 对应 HF 仓库分片文件名中的量化标识（TODO(实测)：待下载后校准）
# shards     : 分片数（-0000{i}-of-0000{n}）
MODES: list[dict[str, Any]] = [
    # ============================================================
    # 1. 极速（最低门槛，小上下文）
    # ============================================================
    {
        "name": "strata/fast",
        "label": "极速",
        "alias": "Strata-Fast",
        "quant_key": "Q2_0",
        "file_tag": "Q2_0",
        "family": "main",
        "shards": 2,
        "ctx": 32768,
        "ctx_min": 8192,
        "ctx_max": 65536,
        "mtp": False,
        "vision": False,
        "precision": "Q2_0 (GGUF)",
        "loader": "strata",
        # vram_required = 真实「显存」需求（Strata 官方门槛 12GB+，AI_SETUP.md L45-48）；
        # 内存需求另由 ram_required_gb 承担（勿混用）
        "vram_required": "12GB",
        "ram_required_gb": 48,
        "disk_required_gb": 66.4,
        "expected_cold_start_s": 90,
        "description": "极速档：Q2_0 权重 + 32K 上下文；显存/内存门槛最低，适合快速对话",
    },
    # ============================================================
    # 2. 均衡（★ 默认：质量与长度平衡，含 MTP + vision）
    # ============================================================
    {
        "name": "strata/balanced",
        "label": "均衡",
        "alias": "Strata-Balanced",
        "quant_key": "IQ2_XS",
        "file_tag": "IQ2_XS",
        "family": "main",
        "shards": 2,
        "ctx": 131072,
        "ctx_min": 32768,
        "ctx_max": 131072,
        "mtp": True,
        "vision": True,
        "precision": "IQ2_XS (GGUF)",
        "loader": "strata",
        "vram_required": "12GB",
        "ram_required_gb": 48,
        "disk_required_gb": 68.0,
        "expected_cold_start_s": 120,
        "description": "均衡档（★默认）：IQ2_XS 权重 + 128K 上下文 + MTP 加速 + vision 多模态",
    },
    # ============================================================
    # 3. 质量（更高量化档，显存门槛更高）
    # ============================================================
    {
        "name": "strata/quality",
        "label": "质量",
        "alias": "Strata-Quality",
        "quant_key": "IQ3_XXS",
        "file_tag": "IQ3_XXS",
        "family": "main",
        "shards": 2,
        "ctx": 131072,
        "ctx_min": 32768,
        "ctx_max": 131072,
        "mtp": False,
        "vision": False,
        "precision": "IQ3_XXS (GGUF)",
        "loader": "strata",
        "vram_required": "16GB",
        "ram_required_gb": 60,
        "disk_required_gb": 75.8,
        "expected_cold_start_s": 150,
        "description": "质量档：IQ3_XXS 权重 + 128K 上下文；画质/推理质量优先，内存≥60GB",
    },
    # ============================================================
    # 4. 长文（超大上下文）
    # ============================================================
    {
        "name": "strata/long",
        "label": "长文",
        "alias": "Strata-Long",
        "quant_key": "IQ2_XS",
        "file_tag": "IQ2_XS",
        "family": "main",
        "shards": 2,
        "ctx": 262144,
        "ctx_min": 131072,
        "ctx_max": 262144,
        "mtp": False,
        "vision": False,
        "precision": "IQ2_XS (GGUF)",
        "loader": "strata",
        "vram_required": "12GB",
        "ram_required_gb": 48,
        "disk_required_gb": 68.0,
        "expected_cold_start_s": 180,
        "description": "长文档档：IQ2_XS 权重 + 256K 上下文；长文本/整本资料处理，KV 显存占用高",
    },
    # ============================================================
    # 5. 编程（Coder 家族权重）
    # ============================================================
    {
        "name": "strata/coder",
        "label": "编程",
        "alias": "Strata-Coder",
        "quant_key": "IQ1_M",
        "file_tag": "IQ1_M",
        "family": "coder",
        "shards": 2,
        "ctx": 32768,
        "ctx_min": 8192,
        "ctx_max": 65536,
        "mtp": False,
        "vision": False,
        "precision": "IQ1_M (GGUF, Coder)",
        "loader": "strata",
        "vram_required": "8GB",
        "ram_required_gb": 32,
        "disk_required_gb": 58.4,
        # 低显存 ⚠️ 提示（T3.2）：仅「低显存档」声明，供 UI 渲染徽标；其余档为 None（默认）
        "low_vram_note": "⚠️ 低显存档（8GB；6GB 级为实验性）：需 RAM ≥32GB + NVMe SSD，显存不足时专家层落盘、速度显著变慢，不承诺速度。",
        "expected_cold_start_s": 90,
        "description": "编程档：Coder 家族 IQ1_M 权重 + 32K 上下文；代码生成/补全，内存门槛最低",
    },
    # ============================================================
    # 6. 精简（低内存环境）
    # ============================================================
    {
        "name": "strata/compact",
        "label": "精简",
        "alias": "Strata-Compact",
        "quant_key": "IQ2_XS",
        "file_tag": "IQ2_XS",
        "family": "main",
        "shards": 2,
        "ctx": 32768,
        "ctx_min": 8192,
        "ctx_max": 65536,
        "mtp": False,
        "vision": False,
        "precision": "IQ2_XS (GGUF)",
        "loader": "strata",
        # vram_required = 真实「显存」需求（Strata 官方门槛 12GB+）；内存需求由 ram_required_gb 承担
        "vram_required": "12GB",
        "ram_required_gb": 48,
        "disk_required_gb": 68.0,
        "expected_cold_start_s": 90,
        "description": "精简档：IQ2_XS 权重（低内存模式）+ 32K 上下文；为内存紧张环境降 KV 占用",
    },
]

# 模式名 → 模式定义 快查表
MODE_LOOKUP: dict[str, dict[str, Any]] = {m["name"]: m for m in MODES}


def get_mode(variant_name: str) -> Optional[dict[str, Any]]:
    """按 variant 名取模式定义（找不到返回 None）。"""
    return MODE_LOOKUP.get(variant_name)


# ===== HF 仓库与文件命名（TODO(实测)：待下载后校准精确文件名）=====

# 主家族仓库
HF_REPO_MAIN = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
# 编程家族仓库
HF_REPO_CODER = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
# mmproj（主家族仓库内；vision 模式用）。
# 注：MTP draft 层不是可下载的 GGUF，由 Strata ``setup.py`` 从 Qwen BF16
#     checkpoint 抽取构建到 ``<data_dir>/mtp/``，故此处不登记文件名。
MMPROJ_FILENAME = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"


def repo_for_family(family: str) -> str:
    """返回家族对应的 HF 仓库 ID。"""
    return HF_REPO_CODER if family == "coder" else HF_REPO_MAIN


def shard_filename(family: str, file_tag: str, index: int, total: int) -> str:
    """返回某模式某个分片的文件名。

    Args:
        family: "main" / "coder"
        file_tag: 量化标识（如 IQ2_XS）
        index: 分片序号（1 起）
        total: 分片总数

    Returns:
        形如 ``Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf``
        （coder 家族前缀为 ``Qwen3.8-Flash-Next-Coder``）
    """
    stem = "Qwen3.8-Flash-Next-Coder" if family == "coder" else "Qwen3.8-Flash-Next"
    return (
        f"{stem}-GSQ-RCO-{file_tag}-"
        f"{index:05d}-of-{total:05d}.gguf"
    )


class ChatInputs(BaseModel):
    """对话推理输入参数（放入 ``InferRequest.inputs``）。

    所有字段均有默认值，网关 / WebUI 按需覆盖。
    ``extra="allow"`` 保证网关透传的未知字段不会导致解析失败。
    """

    model_config = ConfigDict(extra="allow")

    # ===== 输入（messages 为主入口，prompt 为单轮兜底）=====
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="对话消息列表（OpenAI 语义：role/content）",
    )
    prompt: Optional[str] = Field(
        None, description="纯文本提示（无 messages 时包装成单条 user 消息）"
    )

    # ===== 生成参数 =====
    max_tokens: int = Field(512, ge=1, description="最大生成 token 数")
    temperature: float = Field(0.7, ge=0.0, description="采样温度")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="核采样阈值")
    top_k: int = Field(0, ge=0, description="top-k 采样（0 = 不启用）")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="min-p 采样（0 = 不启用）")
    seed: Optional[int] = Field(None, description="随机种子（None = 随机）")
    stop: Optional[list[str]] = Field(
        None,
        description="停止词列表（None / [] = 不启用；网关按 OpenAI 语义常透传 null）",
    )

    # ===== 流式 / 思考 =====
    stream: bool = Field(True, description="是否流式输出（SSE 契约要求默认 True）")
    reasoning_effort: Optional[str] = Field(
        None, description=f"思考强度：{' / '.join(REASONING_EFFORTS)}"
    )

    # ===== 上下文 / 多模态 =====
    context: Optional[int] = Field(
        None, ge=1024, description="覆盖本次请求的上下文长度（一般无需传，由模式决定）"
    )
    vision: bool = Field(False, description="是否启用多模态（需模式自带 mmproj）")

    # ===== 工具调用（agent 场景透传；Strata 原生支持 function calling）=====
    # 平台链路：gateway 传入 inputs.tools → 本引擎原样转发到原生
    # POST /v1/chat/completions（native chat_template 注入 <tools>，frontend 增量
    # 解析 XML 工具调用并回吐 OpenAI 流式 tool_calls）。不透传时模型收不到工具定义，
    # 只能输出文本格式调用，客户端解析 message.tool_calls 崩溃（同 llama 引擎 Hermes 根因）。
    tools: Optional[list[dict[str, Any]]] = Field(
        None,
        description="OpenAI 风格工具定义列表（非空时透传给原生端点，启用 function calling）",
    )
    tool_choice: Optional[Any] = Field(
        None,
        description="工具选择策略：'auto' / 'none' / 'required' 或 "
                    "{'type':'function','function':{'name':...}}（原生支持校验）",
    )

    def resolve_messages(self) -> list[dict[str, Any]]:
        """解析对话消息（messages 优先，其次 prompt 单轮兜底）。

        Returns:
            规范化的 OpenAI 风格消息列表
        """
        if self.messages:
            return self.messages
        if self.prompt:
            return [{"role": "user", "content": self.prompt}]
        return []


class StrataVariantInfo(VariantInfo):
    """Strata 模式能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充模式特有信息：
        - mode              ：模式键（如 "balanced"）
        - mode_label        ：模式中文名（如 "均衡"）
        - ctx_recommended   ：推荐上下文长度
        - ctx_min / ctx_max ：上下文可调范围
        - mtp               ：是否启用 MTP 投机解码
        - vision            ：是否含 mmproj 多模态
        - ram_required_gb   ：内存需求（GB）
        - low_vram_note     ：低显存 ⚠️ 提示（仅低显存档非空，供 UI 渲染）
        - disk_required_gb  ：磁盘体积（GB）
        - recommended       ：是否为本机推荐档（由 engine._recommend() 打标）
        - expected_cold_start_s：预期冷启动耗时（秒）
    """

    model_config = ConfigDict(extra="allow")

    mode: str = Field("", description="模式键（fast/balanced/quality/long/coder/compact）")
    mode_label: str = Field("", description="模式中文名（极速/均衡/…）")
    ctx_recommended: int = Field(131072, description="推荐上下文长度")
    ctx_min: int = Field(8192, description="上下文可调下限")
    ctx_max: int = Field(262144, description="上下文可调上限")
    mtp: bool = Field(False, description="是否启用 MTP 投机解码")
    vision: bool = Field(False, description="是否含 mmproj 多模态")
    ram_required_gb: int = Field(48, description="内存需求（GB）")
    low_vram_note: Optional[str] = Field(
        None, description="低显存 ⚠️ 提示（仅低显存档声明，无则 None）"
    )
    disk_required_gb: float = Field(68.0, description="磁盘体积（GB）")
    recommended: bool = Field(False, description="是否为本机推荐档")
    expected_cold_start_s: int = Field(120, description="预期冷启动耗时（秒）")

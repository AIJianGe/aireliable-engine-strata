"""Strata LLM 推理引擎核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 ``StrataProcessManager`` 的加载 / 推理 / 卸载 / 模式切换流程。

8 端点契约：
    - /capabilities  返回 6 个命名模式（含推荐标记 recommended）
    - /load          校验 bins + GGUF 分片（**只校验不下载**）→ 冷启动 strata.exe
    - /unload        停止子进程 + 释放状态
    - /infer         解析 ChatInputs → 转发内部 OpenAI 端点 → SSE 推送
    - /status        返回引擎状态（模式 / 内部端口 / pid / 硬件画像）
    - /lora/load     v1 不支持 LoRA（预留接口）
    - /lora/unload   v1 不支持 LoRA（预留接口）

增强端点（server.py 注册）：
    - /hardware      硬件探针 + 推荐模式（_probe_hardware / _recommend）
    - /mode/switch   冷切换模式（停 → 改写 → 冷启动，失败回滚）

任务类型：chat（多轮对话生成）。

⚠️ 关键契约：
    1. ``load`` 只校验不下载（缺失抛 RuntimeError，信息含路径 + 仓库 + 下载指引）
    2. 推理失败**必须真实发 stage=error 帧**（R25），不得静默
    3. ``--expert-cache`` 锁 auto（R17）；MTP spec 默认 4（R19）
    4. LoRA 不支持；音色管理（data/voice_library.db / data/voices/）零触碰
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.sse import make_error_frame, make_final_frame, make_frame
from aireliable_engine.state_machine import RuntimeState

from .config import EngineConfig
from .model_scanner import ModelScanner
from .schemas import (
    CAPABILITY,
    MODES,
    TASK_TYPE,
    ChatInputs,
    StrataVariantInfo,
    get_mode,
)
from .strata_process import StrataProcessManager

logger = logging.getLogger(__name__)


def _accumulate_tool_calls(acc: dict[int, dict], tc_list: list[Any]) -> None:
    """把 OpenAI 流式 ``delta.tool_calls`` 增量按 index 合并进 ``acc``。

    合并规则（对齐 OpenAI / llama-server 流式语义）：
        - ``id``：首个增量整体给出，覆盖式
        - ``function.name``：首个增量整体给出，覆盖式
        - ``function.arguments``：分片到达，拼接式

    无法解析的 index 归 0；非 dict 项忽略。

    Args:
        acc: index → ``{"id","type","function":{"name","arguments"}}`` 聚合容器
        tc_list: 本 chunk 的 ``tool_calls`` 增量列表
    """
    for tc in tc_list or []:
        if not isinstance(tc, dict):
            continue
        idx = tc.get("index", 0)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = 0
        acc_i = acc.setdefault(idx, {
            "id": "", "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if tc.get("id"):
            acc_i["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            acc_i["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            acc_i["function"]["arguments"] += fn["arguments"]


class StrataEngine(CustomEngine):
    """Strata LLM 推理引擎（integrated 范式）。

    6 个命名模式（见 ``schemas.MODES``），每个模式 = 量化档 + 上下文 + MTP/vision。
    真实推理由外部二进制 ``strata.exe`` 承担，本类只做：
        - 8 端点契约
        - 模式冷切换（停 → 改写 → 冷启动，失败回滚）
        - 硬件探针 + 推荐
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # ===== 核心组件 =====
        self.process = StrataProcessManager(config, logger_=logger)
        # 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 启动时扫描（同步 fallback，异步扫描在 lifespan 中触发）
        self._available_variants: list[StrataVariantInfo] = []
        self._refresh_scan()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        _manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant = (
            self._read_default_variant_from_manifest(_manifest_path) or "strata/balanced"
        )

        # 任务类型（任务中心筛选语义）
        self.task_type = TASK_TYPE

        # 已加载模式（与 process.current_mode 同步）
        self.loaded_mode: Optional[str] = None
        # 硬件探针缓存
        self._hardware: Optional[dict[str, Any]] = None

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新资源扫描结果（同步 fallback，用于 __init__ 阶段）。"""
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """方案 B：异步刷新扫描结果（优先平台 API），lifespan startup 阶段调用。"""
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ============ 硬件探针 + 推荐（阶段 4）============

    def _probe_hardware(self) -> dict[str, Any]:
        """硬件探针：GPU / 内存 / CPU 特性 / 磁盘空间。

        任何子项失败都降级为「无法检测」，**不阻断启动**。
        结果缓存到 ``self._hardware``（供 /capabilities 推荐与 /hardware 端点复用）。
        """
        info: dict[str, Any] = {
            "gpu": None,
            "vram_total_gb": None,
            "compute_cap": None,
            "driver_version": None,
            "ram_total_gb": None,
            "ram_available_gb": None,
            "cpu_features": [],
            "disk_free_gb": None,
            "probe_ok": False,
        }

        # ----- GPU（nvidia-smi）-----
        try:
            import subprocess

            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
                creationflags=(0x08000000 if __import__("sys").platform == "win32" else 0),
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                if len(parts) >= 4:
                    info["gpu"] = parts[0]
                    info["vram_total_gb"] = round(float(parts[1]) / 1024.0, 1)
                    info["compute_cap"] = parts[2]
                    info["driver_version"] = parts[3]
        except Exception as e:  # noqa: BLE001
            logger.info("nvidia-smi 探针不可用（降级）: %s", e)

        # ----- 内存 / 磁盘 / CPU 特性（psutil）-----
        try:
            import psutil

            info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
            info["ram_available_gb"] = round(
                psutil.virtual_memory().available / (1024 ** 3), 1
            )
            try:
                info["disk_free_gb"] = round(
                    psutil.disk_usage(str(self.config.models_root)).free / (1024 ** 3), 1
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.info("psutil 探针不可用（降级）: %s", e)

        # AVX2 / AVX-512（仅 Linux 可从 /proc/cpuinfo 读；Windows 用 platform 降级）
        try:
            import platform as _pf

            if _pf.system() == "Linux":
                cpuinfo = Path("/proc/cpuinfo")
                if cpuinfo.exists():
                    text = cpuinfo.read_text(errors="ignore")
                    if "avx512" in text:
                        info["cpu_features"].append("avx512")
                    if "avx2" in text:
                        info["cpu_features"].append("avx2")
        except Exception:  # noqa: BLE001
            pass

        info["probe_ok"] = bool(info["gpu"] or info["ram_total_gb"])
        self._hardware = info
        logger.info(
            "硬件探针：gpu=%s, vram=%sGB, cc=%s, ram=%sGB, disk_free=%sGB",
            info["gpu"], info["vram_total_gb"], info["compute_cap"],
            info["ram_total_gb"], info["disk_free_gb"],
        )
        return info

    def get_hardware(self) -> dict[str, Any]:
        """返回硬件画像（首次调用时探针，之后复用缓存）。"""
        if self._hardware is None:
            self._probe_hardware()
        return self._hardware or {}

    def _recommend(self) -> str:
        """显存分层推荐算法，返回推荐模式 variant 名。

        依据修正后的 vram_required 语义（fast/balanced/long/compact=12GB、
        quality=16GB、coder=8GB；内存门槛由 ram_required_gb 承担）分层：

            vram≥16 且 ram≥60 → quality
            vram≥12 且 ram≥48 → balanced（★ 默认，带 vision + MTP）
            vram≥8  且 ram≥32 → coder
            ram≥48（显存<8）  → fast（Q2_0 更省显存 + 专家自动 CPU 回退）
            ram≥32            → coder
            否则              → compact

        探针失败 → 保守 balanced。低显存（<12GB）时置 ``self._recommend_note``
        供 /hardware 与 UI 提示「实验性 / 慢 ⚠️」。

        注：agent 长上下文场景可手动选 ``strata/long``（256K，无 MTP），
            不在此自动推荐（默认保持 ★balanced）。
        """
        hw = self.get_hardware()
        self._recommend_note: Optional[str] = None
        if not hw.get("probe_ok"):
            logger.info("推荐算法：探针不可用 → 保守推荐 strata/balanced")
            return "strata/balanced"

        ram = hw.get("ram_total_gb") or 0
        vram = hw.get("vram_total_gb") or 0

        # 低显存提示（不影响选档，仅提示用户预期）
        if vram and vram < 8:
            self._recommend_note = (
                "⚠️ 实验性：显存 <8GB。官方不背书（12GB 为推荐下限），"
                "仅 coder 档（IQ1_M）有机会运行且速度很慢；需 NVMe + ≥32GB 内存，"
                "专家缓存极小、decode 抖动明显。"
            )
        elif vram and vram < 12:
            self._recommend_note = (
                "⚠️ 低显存：8–12GB。建议 coder 档（IQ1_M，专家数减半）；"
                "其余档标称需 12GB，专家缓存小、速度偏慢。"
            )

        # 门禁分层 —— 由高到低挑选「显存与内存都满足」的最高档
        if vram >= 16 and ram >= 60:
            logger.info("推荐算法：RAM=%.0fGB/VRAM=%.0fGB → strata/quality", ram, vram)
            return "strata/quality"
        if vram >= 12 and ram >= 48:
            logger.info("推荐算法：RAM=%.0fGB/VRAM=%.0fGB → strata/balanced", ram, vram)
            return "strata/balanced"
        if vram >= 8 and ram >= 32:
            logger.info("推荐算法：RAM=%.0fGB/VRAM=%.0fGB → strata/coder", ram, vram)
            return "strata/coder"
        if ram >= 48:
            logger.info("推荐算法：RAM=%.0fGB/VRAM=%.0fGB（显存<8）→ strata/fast", ram, vram)
            return "strata/fast"
        if ram >= 32:
            logger.info("推荐算法：RAM=%.0fGB → strata/coder", ram)
            return "strata/coder"

        logger.info("推荐算法：RAM=%.0fGB 低于全部档位门槛 → 保守 strata/compact", ram)
        return "strata/compact"

    # ============ 完整性校验（只校验不下载）============

    def _check_bins_integrity(self) -> None:
        """校验 strata-bins 可执行文件存在（缺失抛 RuntimeError + 下载指引）。"""
        exe = self.config.strata_exe
        if exe.exists():
            return
        raise RuntimeError(
            f"Strata 可执行文件缺失：{exe}\n"
            f"  · 二进制目录：{self.config.bins_dir}\n"
            f"  · 期望文件：{exe.name}（Windows）/ strata（Linux）\n"
            f"  · 官方来源：https://github.com/Niko1221/Strata（Release v0.1.45）\n"
            f"请到引擎详情页 → 模型管理 → 下载 strata-bins，"
            f"或手动放置到上述二进制目录后重试。"
        )

    def _check_models_dir_integrity(self, variant: str) -> None:
        """校验该模式的 GGUF 分片（+ vision 模式的 mmproj）存在。

        **只校验不下载**：缺失时抛 RuntimeError，信息含
        缺失清单 + 本地路径 + HF 仓库 ID + 模型管理界面下载指引。
        """
        if not self.config.model_dir.exists():
            raise RuntimeError(
                f"模型目录不存在：{self.config.model_dir}\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」，"
                f"选择镜像源后下载（HF 仓库：ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF）"
            )

        missing: list[str] = []
        for p in self.scanner.expected_shard_paths(variant):
            if not p.exists():
                missing.append(str(p))
        for p in self.scanner.expected_extra_paths(variant):
            if not p.exists():
                missing.append(str(p))

        if missing:
            mode_def = self.scanner.get_variant_definition(variant) or {}
            repo = (
                "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
                if mode_def.get("family") == "coder"
                else "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
            )
            raise RuntimeError(
                f"模型文件不完整（模式 {variant}），缺失 {len(missing)} 个文件：\n"
                f"  " + "\n  ".join(missing) + "\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"本地路径：{self.config.model_dir}\n"
                f"HuggingFace 仓库：{repo}"
            )

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回 6 个命名模式（含推荐标记）。"""
        recommended = self._recommend()

        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(StrataVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                mode=v.mode,
                mode_label=v.mode_label,
                ctx_recommended=v.ctx_recommended,
                ctx_min=v.ctx_min,
                ctx_max=v.ctx_max,
                mtp=v.mtp,
                vision=v.vision,
                ram_required_gb=v.ram_required_gb,
                low_vram_note=v.low_vram_note,
                disk_required_gb=v.disk_required_gb,
                recommended=(v.name == recommended),
                expected_cold_start_s=v.expected_cold_start_s,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=[],  # v1 不支持 LoRA
            supported_sizes=[],
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema
            input_schema=ChatInputs.model_json_schema(),
            # 单能力引擎：capabilities 列表退化为 [capability]
            capabilities=[self.capability],
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load：校验 bins + GGUF 分片（**只校验不下载**）→ 冷启动。

        流程：
            1. variant（模式）缺省兜底
            2. bins 完整性校验
            3. 该模式 GGUF 分片（+ vision 的 mmproj）完整性校验
            4. 设备解析（不留 "auto" 歧义）
            5. 冷启动 strata.exe 并等待就绪
        """
        # variant 缺省时回退（SDK LoadRequest 契约）
        if not variant:
            variant = self.default_variant or next(
                (v.name for v in self._available_variants), None
            )
        if not variant:
            raise RuntimeError(
                "未指定模式，且 manifest 无 default_variant / 无可用模式，无法加载"
            )

        mode_def = self.scanner.get_variant_definition(variant)
        if not mode_def:
            raise RuntimeError(
                f"模式定义缺失（manifest 与 schemas.MODES 不一致?）: {variant}"
            )

        # 只校验不下载
        self._check_bins_integrity()
        self._check_models_dir_integrity(variant)

        device = self.config.resolve_device()
        ctx = int(mode_def.get("ctx", 131072))
        # MTP：模式声明 mtp 时用 config.spec（默认 4），否则 0（关闭，R19）
        spec = self.config.spec if mode_def.get("mtp") else 0
        vision = bool(mode_def.get("vision", False))

        start_time = time.time()
        # 已加载不同模式 → 走冷切换编排（含回滚）
        if self.process.is_running and self.process.current_mode != variant:
            try:
                await self.process.switch_mode(
                    old_mode=self.process.current_mode,
                    new_mode=variant, ctx=ctx, spec=spec, vision=vision, device=device,
                )
            except Exception as e:
                logger.exception("模式冷切换失败: %s → %s", self.process.current_mode, variant)
                raise RuntimeError(f"模式切换失败: {e}") from e
        else:
            await self.process.start(
                mode_name=variant, ctx=ctx, spec=spec, vision=vision, device=device
            )

        load_time = time.time() - start_time
        self.loaded_mode = variant
        logger.info(
            "加载就绪: mode=%s, ctx=%s, spec=%s, vision=%s, device=%s, internal_port=%s, load_time=%.2fs",
            variant, ctx, spec, vision, device, self.process.internal_port, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used="外部进程持有",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：停止子进程 + 释放状态。

        即使未加载（如 ERROR 状态下），也确保状态机回到 IDLE，避免卡死。
        """
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:  # noqa: BLE001
                logger.warning("unload()：reset 状态机失败: %s", e)

        if self.process.is_running:
            await self.process.stop()

        self.loaded_mode = None
        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:  # noqa: BLE001
                logger.warning("unload()：状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成（strata.exe 子进程已停止，模式状态已清空）")
        return UnloadResponse(state="idle", vram_freed="外部进程已退出")

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：对话生成（SSE 流式推送）。

        流程：
            1. 生成 / 复用 task_id
            2. 解析 ChatInputs（messages 主入口，prompt 单轮兜底）
            3. 处理前自动加载（模式变化时先冷切换）
            4. 转发内部 OpenAI 兼容端点 /v1/chat/completions
            5. 增量解析 SSE → 推送 stage/delta 帧
            6. 完成推送 completed 帧（含 text / text_url / usage）

        错误处理：内部捕获异常，推送 **stage=error** 帧后 return（R25），
        **不重新抛出**（避免 BaseEngine.infer 把状态置为 ERROR）。
        """
        started = time.time()
        task_id = request.task_id or f"strata_{uuid.uuid4().hex[:16]}"

        # ----- 1. 解析输入 -----
        try:
            inputs = ChatInputs(**(request.inputs or {}))
        except Exception as e:
            logger.error("ChatInputs 解析失败: %s", e)
            await stream.put(make_error_frame(f"输入参数无效: {e}", "invalid_inputs"))
            return

        messages = inputs.resolve_messages()
        if not messages:
            await stream.put(make_error_frame(
                "缺少输入：请通过 inputs.messages 传入对话消息，或 inputs.prompt 传入单轮提示",
                "missing_input",
            ))
            return

        # ----- 2. 处理前自动加载 -----
        requested_mode = request.variant or self.loaded_mode or self.default_variant
        if requested_mode and (not self.process.is_running or self.process.current_mode != requested_mode):
            logger.info(
                "处理前自动加载: 已加载=%s, 请求=%s → 冷启动/切换",
                self.process.current_mode, requested_mode,
            )
            await stream.put(make_frame(0.0, "loading", message=f"模式加载中（{requested_mode}）…"))
            try:
                await self._load_impl(requested_mode)
            except Exception as e:
                logger.exception("处理前自动加载失败: mode=%s", requested_mode)
                await stream.put(make_error_frame(f"模式加载失败: {e}", "load_failed"))
                return

        # ----- 3. 转发内部 OpenAI 端点 -----
        if not self.process.internal_url:
            await stream.put(make_error_frame("Strata 子进程未就绪（internal_url 为空）", "not_ready"))
            return

        url = f"{self.process.internal_url}/v1/chat/completions"
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(inputs.max_tokens),
            "temperature": float(inputs.temperature),
            "top_p": float(inputs.top_p),
            "stream": True,
        }
        if inputs.top_k:
            body["top_k"] = int(inputs.top_k)
        if inputs.min_p:
            body["min_p"] = float(inputs.min_p)
        if inputs.seed is not None:
            body["seed"] = int(inputs.seed)
        if inputs.stop:
            body["stop"] = list(inputs.stop)
        if inputs.reasoning_effort:
            body["reasoning_effort"] = inputs.reasoning_effort
        # 工具调用透传（agent 场景）：tools / tool_choice 直达原生端点。
        # 不透传时模型收不到工具定义，只能输出文本格式调用 → 客户端解析崩溃。
        if inputs.tools:
            body["tools"] = inputs.tools
            logger.info(
                "/infer 工具调用透传: tools=%d 个, tool_choice=%s",
                len(inputs.tools), inputs.tool_choice,
            )
        if inputs.tool_choice is not None:
            body["tool_choice"] = inputs.tool_choice

        await stream.put(make_frame(0.1, "generating", message="开始生成…"))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: Optional[str] = None
        usage: dict[str, Any] = {}
        first_token_at: Optional[float] = None
        # 工具调用聚合器：index → {"id","type","function":{"name","arguments"}}
        # 流式增量按 index 合并（id/name 覆盖式、arguments 拼接式），完成帧输出完整列表
        tool_calls_acc: dict[int, dict] = {}

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout, connect=15.0),
                trust_env=False,
            ) as client:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code >= 400:
                        raw = await resp.aread()
                        raise RuntimeError(
                            f"Strata 子进程返回 HTTP {resp.status_code}: "
                            f"{raw.decode('utf-8', errors='replace')[:500]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        delta_text = delta.get("content") or ""
                        delta_reason = delta.get("reasoning_content") or ""
                        tc_delta = delta.get("tool_calls") or []
                        if delta_text:
                            if first_token_at is None:
                                first_token_at = time.time()
                            text_parts.append(delta_text)
                        if delta_reason:
                            reasoning_parts.append(delta_reason)
                        # 工具调用增量：按 index 聚合（id/name 覆盖式、arguments 拼接式）
                        _accumulate_tool_calls(tool_calls_acc, tc_delta)
                        # 推一帧（content / reasoning / tool_calls 增量）。
                        # 契约：content 走 delta 顶层；思维链走 extra.reasoning_content
                        # （gateway 流式读 frame.extra.reasoning_content）；工具增量走
                        # 顶层 tool_calls（gateway 原样转发为 OpenAI delta.tool_calls）。
                        if delta_text or delta_reason or tc_delta:
                            frame = make_frame(
                                0.5, "generating", delta=delta_text,
                                message="", text="".join(text_parts),
                            )
                            if delta_reason:
                                frame["extra"] = {"reasoning_content": delta_reason}
                            if tc_delta:
                                frame["tool_calls"] = tc_delta
                            await stream.put(frame)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        if chunk.get("usage"):
                            usage = chunk["usage"]
        except Exception as e:
            logger.exception("Strata 推理失败: task_id=%s", task_id)
            await stream.put(make_error_frame(f"推理失败: {e}", "infer_failed"))
            return

        # ----- 4. 产物（文本落盘 + URL）-----
        text = "".join(text_parts)
        text_path: Optional[str] = None
        text_url: Optional[str] = None
        if text:
            try:
                out_path = self.get_output_path("text", f"{task_id}.txt")
                out_path.write_text(text, encoding="utf-8")
                text_path = str(out_path)
                rel = out_path.relative_to(self.output_root)
                text_url = "/output/" + str(rel).replace("\\", "/")
            except Exception as e:  # noqa: BLE001
                logger.warning("文本产物落盘失败: %s", e)

        elapsed = time.time() - started
        completion_tokens = (usage or {}).get("completion_tokens")
        timing = {
            "total": elapsed,
            "ttft": (first_token_at - started) if first_token_at else None,
        }
        if completion_tokens and elapsed > 0:
            timing["tok_per_s"] = round(completion_tokens / elapsed, 2)

        result: dict[str, Any] = {
            "text": text,
            "text_path": text_path,
            "text_url": text_url,
            "finish_reason": finish_reason,
            "reasoning_content": "".join(reasoning_parts) or None,
            "completion_tokens": completion_tokens,
            "usage": usage,
            "mode": self.loaded_mode,
            "variant": self.loaded_mode,
            "task_id": task_id,
        }
        # 工具调用透传：聚合列表放入完成帧 result（gateway 同步模式读
        # result.tool_calls 构造 message.tool_calls；流式模式读增量帧）
        merged_tool_calls: list[dict] = [
            tool_calls_acc[i] for i in sorted(tool_calls_acc)
        ]
        if merged_tool_calls:
            result["tool_calls"] = merged_tool_calls
            logger.info(
                "工具调用完成: %d 个, finish_reason=%s, names=%s",
                len(merged_tool_calls), finish_reason,
                [tc["function"].get("name", "") for tc in merged_tool_calls],
            )

        logger.info(
            "对话完成: task_id=%s, mode=%s, tokens=%s, elapsed=%.2fs, ttft=%.2fs",
            task_id, self.loaded_mode, completion_tokens, elapsed,
            timing.get("ttft") if timing.get("ttft") is not None else -1,
        )
        await stream.put(make_final_frame(result, timing))

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态（模式 / 内部端口 / pid / 硬件画像）。"""
        hw = self.get_hardware()
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.loaded_mode,
            vram_used="外部进程持有",
            loaded_loras=[],  # v1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "mode": self.process.current_mode or self.loaded_mode,
                "mode_label": (get_mode(self.process.current_mode or "") or {}).get("label", ""),
                "ctx": self.process.current_ctx,
                "internal_port": self.process.internal_port,
                "internal_url": self.process.internal_url,
                "pid": self.process.pid,
                "is_running": self.process.is_running,
                "spec": self.config.spec,
                "device": self.config.resolve_device(),
                "hardware": hw,
                "recommended_mode": self._recommend(),
                "task_type": self.task_type,
                "capability": CAPABILITY,
                "model_dir": str(self.config.model_dir),
                "bins_dir": str(self.config.bins_dir),
                "strata_data_dir": str(self.config.strata_data_dir),
                "scan_path": self.scanner.last_scan_mode,
                "available_variants": [
                    {
                        "name": v.name,
                        "mode": v.mode,
                        "mode_label": v.mode_label,
                        "available": v.available,
                        "ctx_recommended": v.ctx_recommended,
                        "mtp": v.mtp,
                        "vision": v.vision,
                        "ram_required_gb": v.ram_required_gb,
                        "disk_required_gb": v.disk_required_gb,
                    }
                    for v in self._available_variants
                ],
            },
        )

    # ============ 模式冷切换（供 POST /mode/switch）============

    async def switch_mode(self, new_mode: str) -> dict[str, Any]:
        """冷切换模式（停 → 改写 → 冷启动，失败回滚）。

        Raises:
            RuntimeError: 新模式不存在 / 启动失败（已尝试回滚）
        """
        mode_def = get_mode(new_mode)
        if not mode_def:
            raise RuntimeError(
                f"未知模式: {new_mode}（可用: {[m['name'] for m in MODES]}）"
            )
        await self._check_bins_integrity_safe()
        self._check_models_dir_integrity(new_mode)

        device = self.config.resolve_device()
        spec = self.config.spec if mode_def.get("mtp") else 0
        await self.process.switch_mode(
            old_mode=self.process.current_mode, new_mode=new_mode,
            ctx=int(mode_def.get("ctx", 131072)), spec=spec,
            vision=bool(mode_def.get("vision", False)), device=device,
        )
        self.loaded_mode = new_mode
        return {
            "mode": new_mode,
            "ctx": self.process.current_ctx,
            "spec": spec,
            "vision": bool(mode_def.get("vision", False)),
            "internal_port": self.process.internal_port,
            "pid": self.process.pid,
        }

    async def _check_bins_integrity_safe(self) -> None:
        """bins 校验的异步包装（保持接口一致，便于后续扩展异步检测）。"""
        self._check_bins_integrity()

    # ============ LoRA（v1 预留）============

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：Strata v1 不支持 LoRA（预留接口）。"""
        raise RuntimeError("Strata LLM v1 不支持 LoRA（预留接口，后续版本扩展）")

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：Strata v1 不支持 LoRA（预留接口）。"""
        raise RuntimeError("Strata LLM v1 不支持 LoRA（预留接口，后续版本扩展）")

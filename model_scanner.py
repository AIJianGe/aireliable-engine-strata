"""Strata LLM 模型/模式扫描器（6 模式 variant 版）。

扫描 ``models/llm/checkpoints/strata/`` 目录，识别可用的命名模式。

6 模式设计（见 ``schemas.MODES``）：
    - strata/fast     极速 / q2_0     / 32K
    - strata/balanced 均衡 / IQ2_XS    / 128K（★默认，含 MTP + vision）
    - strata/quality  质量 / IQ3_XXS   / 128K
    - strata/long     长文 / IQ2_XS    / 256K
    - strata/coder    编程 / IQ1_M     / 32K（Coder 家族）
    - strata/compact  精简 / IQ2_XS    / 32K（低内存）

方案 B 约定（与 video_up_enlarge 一致）：
    优先调用平台 API（``engine.get_variants()``）获取 variant 可用性，
    平台不可达时 fallback 到本地扫描；两条路径都记录 ``last_scan_mode``，
    并输出「variant 扫描路径：平台 API / 本地 fallback（原因：…）」。

本地 fallback 检查策略（project-rules §第二部分 5）：
    - 只检查「该模式的**任一 GGUF 分片**是否落盘」这一标志文件存在性，
      **不校验大小、不要求全部分片齐全**
      （下载一半时也不能让模式从 UI 消失，否则用户失去下载入口）
    - 完整性校验由平台 API（key_files 机制）或 ``engine._load_impl`` 兜底

模型目录结构（与 manifest.yaml model_files.target_dir 一致）：
    models/llm/checkpoints/strata/
    ├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
    ├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
    ├── Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf   （均衡/长文/精简共用）
    ├── Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf
    ├── Qwen3.8-Flash-Next-Coder-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf
    └── mmproj-Qwen3.8-Flash-Next-BF16.gguf                         （vision）

注：MTP draft 层不是模型目录里的 GGUF，由 Strata ``setup.py`` 从 Qwen BF16
    checkpoint 抽取构建到 ``<strata_data_dir>/mtp/``，不经本目录扫描/校验。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import (
    MODES,
    MMPROJ_FILENAME,
    StrataVariantInfo,
    shard_filename,
)

logger = logging.getLogger(__name__)

# 已知模式定义（单一真相源来自 schemas.MODES）
_KNOWN_VARIANTS: list[dict] = MODES


class ModelScanner:
    """Strata 模式扫描器（6 模式 variant 版）。

    方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/）
            model_dir: 本引擎模型目录（llm/checkpoints/strata/）
            engine: 可选的 BaseEngine 实例（方案 B 注入）。
                传入时优先调用平台 API 检测 variant；为 None 时走本地扫描。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)
        # 最近一次扫描走的路径：'platform_api' / 'local_fallback'
        self.last_scan_mode: Optional[str] = None
        # 最近一次 fallback 原因（便于日志与排障）
        self.last_scan_reason: Optional[str] = None

    # ===== 扫描入口 =====

    async def scan_variants_async(self) -> list[StrataVariantInfo]:
        """方案 B：异步扫描，优先调用平台 API。"""
        if self.engine is not None:
            try:
                platform_variants = await self.engine.get_variants()
            except Exception as e:
                platform_variants = None
                self.last_scan_reason = f"平台 API 异常: {e}"
                self._logger.warning(
                    "variant 扫描路径：本地 fallback（原因：平台 API 异常 %s，扫描路径 %s）",
                    e, self.model_dir,
                )
            if platform_variants is not None:
                self.last_scan_mode = "platform_api"
                core_url = getattr(self.engine, "core_url", None)
                self._logger.info(
                    "variant 扫描路径：平台 API（AIRELIABLE_CORE_URL=%s）；共 %d 个模式",
                    core_url or "(未设置)", len(platform_variants),
                )
                return self._merge_with_known(platform_variants)
            if self.last_scan_reason is None:
                self.last_scan_reason = "平台 API 返回空"

        # 无 engine / 平台不可达 → 本地 fallback
        if self.last_scan_mode != "platform_api":
            self.last_scan_mode = "local_fallback"
            self._logger.info(
                "variant 扫描路径：本地 fallback（原因：%s，扫描路径 %s）",
                self.last_scan_reason or "未提供平台 API 句柄", self.model_dir,
            )
        return self._scan_locally()

    def scan_variants(self) -> list[StrataVariantInfo]:
        """同步扫描接口（fallback 本地扫描）。"""
        self.last_scan_mode = "local_fallback"
        self.last_scan_reason = self.last_scan_reason or "同步扫描调用"
        return self._scan_locally()

    # ===== 平台 API 合并 =====

    def _merge_with_known(self, platform_variants: list[dict]) -> list[StrataVariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS（schemas.MODES）模式特有字段。"""
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[StrataVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            if not kv:
                # 平台声明了引擎不认识（或已下线）的 variant：跳过，避免误报可用
                self._logger.debug("忽略未知 variant（不在 MODES 中）: %s", v_name)
                continue

            available = bool(pv.get("available", False))
            result.append(self._build_variant_info(kv, available, platform=pv))
            if available:
                self._logger.info("发现模式变体: %s (%s)", v_name, kv.get("label", ""))

        if not any(v.available for v in result):
            self._warn_nothing_found()
        return result

    # ===== 本地扫描 =====

    def _scan_locally(self) -> list[StrataVariantInfo]:
        """fallback 本地扫描（只检查标志文件「任一 GGUF 分片」存在性，不校验大小）。"""
        capabilities: list[StrataVariantInfo] = []

        if not self.model_dir.exists():
            self._logger.warning("模型目录不存在: %s", self.model_dir)
            self._warn_nothing_found()
            return capabilities

        for mode_def in _KNOWN_VARIANTS:
            ready = self._check_mode_ready(mode_def)
            if not ready:
                self._logger.debug(
                    "模式 %s 尚未就位（缺 %s 分片）: %s",
                    mode_def["name"], mode_def.get("file_tag", ""), self.model_dir,
                )
                continue

            cap = self._build_variant_info(mode_def, True)
            capabilities.append(cap)
            self._logger.info(
                "发现模式变体: %s (%s, quant=%s)",
                cap.name, mode_def.get("label", ""), mode_def.get("quant_key", ""),
            )

        if not capabilities:
            self._warn_nothing_found()

        return capabilities

    def _check_mode_ready(self, mode_def: dict) -> bool:
        """检查模式权重是否就位（ANY 语义：该量化档的**任一**分片已落盘即视为可见）。"""
        for name in self.expected_shard_names(mode_def):
            if (self.model_dir / name).exists():
                return True
        return False

    # ===== variant 构造 =====

    def _build_variant_info(
        self,
        mode_def: dict,
        available: bool,
        platform: Optional[dict] = None,
    ) -> StrataVariantInfo:
        """由模式定义（+ 可选平台数据）构造 StrataVariantInfo。"""
        pv = platform or {}
        return StrataVariantInfo(
            name=mode_def["name"],
            base_model=pv.get("base_model", "Qwen3.8-Flash-Next"),
            precision=pv.get("precision", mode_def.get("precision", "IQ2_XS (GGUF)")),
            loader=pv.get("loader", mode_def.get("loader", "strata")),
            vram_required=pv.get("vram_required", mode_def.get("vram_required", "12GB")),
            available=available,
            mode=mode_def["name"].split("/", 1)[-1],
            mode_label=mode_def.get("label", ""),
            ctx_recommended=mode_def.get("ctx", 131072),
            ctx_min=mode_def.get("ctx_min", 8192),
            ctx_max=mode_def.get("ctx_max", 262144),
            mtp=bool(mode_def.get("mtp", False)),
            vision=bool(mode_def.get("vision", False)),
            ram_required_gb=int(mode_def.get("ram_required_gb", 48)),
            low_vram_note=pv.get("low_vram_note", mode_def.get("low_vram_note")),
            disk_required_gb=float(mode_def.get("disk_required_gb", 68.0)),
            recommended=bool(mode_def.get("name") == "strata/balanced"),
            expected_cold_start_s=int(mode_def.get("expected_cold_start_s", 120)),
        )

    # ===== 路径与定义解析 =====

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据模式名解析权重目录（本引擎 6 模式共用同一目录）。"""
        for mode_def in _KNOWN_VARIANTS:
            if mode_def["name"] == variant_name:
                return self.model_dir if self.model_dir.exists() else None
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取模式的完整定义（供 ``engine._load_impl`` 生成 json / 校验文件）。"""
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None

    def expected_shard_names(self, mode_def: dict) -> list[str]:
        """返回该模式期望的 GGUF 分片文件名列表。"""
        family = mode_def.get("family", "main")
        tag = mode_def.get("file_tag", "")
        total = int(mode_def.get("shards", 2))
        if not tag:
            return []
        return [
            shard_filename(family, tag, i, total) for i in range(1, total + 1)
        ]

    def expected_shard_paths(self, variant_name: str) -> list[Path]:
        """返回该模式期望的 GGUF 分片绝对路径列表（供 load 校验）。"""
        mode_def = self.get_variant_definition(variant_name)
        if not mode_def:
            return []
        return [self.model_dir / n for n in self.expected_shard_names(mode_def)]

    def expected_extra_paths(self, variant_name: str) -> list[Path]:
        """返回该模式额外依赖的**模型目录内**文件绝对路径列表。

        仅 mmproj（vision 模式）落在 ``models/llm/checkpoints/strata/``；
        MTP draft 层由 Strata ``setup.py`` 从 Qwen BF16 checkpoint 抽取并构建到
        ``<strata_data_dir>/mtp/``（不在模型目录），故不经此校验。
        """
        mode_def = self.get_variant_definition(variant_name)
        if not mode_def:
            return []
        extras: list[Path] = []
        if mode_def.get("vision"):
            extras.append(self.model_dir / MMPROJ_FILENAME)
        return extras

    # ===== 诊断 =====

    def _warn_nothing_found(self) -> None:
        """扫描无结果时的统一告警（含具体路径 + 仓库 ID + 下载指引）。"""
        self._logger.warning(
            "未在 %s 中找到任何可用的 chat 模式权重，"
            "请到「模型管理」界面下载 Qwen3.8-Flash-Next GGUF 分片"
            "（HuggingFace 仓库：ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF，"
            "编程档另见 …-Coder-GSQ-RCO-GGUF）",
            self.model_dir,
        )

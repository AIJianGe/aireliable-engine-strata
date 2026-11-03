"""Strata LLM 引擎配置。

从环境变量 + 命令行参数加载引擎运行所需的全部配置。

设计参考（与 video_up_enlarge_98782fcd 的双源真相消除策略一致）：
    - MODEL_SUBDIR 默认读环境变量 ``AIRELIABLE_MODEL_LAYOUT_BASE``
      （由 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：命令行参数 > 环境变量 > 硬编码 fallback

命名与端口（ANALYSIS-TASKS/61-* 分析文档 §3.3.6 / §3.5）：
    - engine_id  : strata_llm_7c4f2b91（目录 / manifest / config 三同名契约）
    - engine_name: Strata
    - capability : chat
    - 端口       : 7010（7001-7008 已占用、7009 预留、7010 空闲）
    - 模型布局   : llm/checkpoints/strata/

⚠️ 与 llama_cpp_9cdb76f8 的两处有意差异：
    1. 本引擎的 GGUF 模型按「命名模式」而非「逐文件 variant」组织，
       模式决定 量化档 + 上下文长度 + 是否启用 MTP draft / vision（见 schemas.MODES）
    2. 外部运行时是 ``strata.exe``（STRATA_BINS_DIR 指定），
       其工作数据目录由 ``STRATA_DATA_DIR`` 指定——必须与模型目录解耦，
       否则 Strata 首次启动会试图二次下载 ~70GB 权重（T5.5 规避点）

关键环境变量：
    STRATA_* 由 config/engines/strata_llm_7c4f2b91.yaml 的
    ``deployment.local_config.env`` 自动合并注入；AIRELIABLE_* 由平台 core 注入
    （均无需改 core）：
    AIRELIABLE_CORE_URL           平台 API 地址（http://127.0.0.1:8001）
    AIRELIABLE_MODELS_ROOT        models 根目录
    AIRELIABLE_MODEL_LAYOUT_BASE  当前引擎的 model_layout.base（llm/checkpoints/strata/）
    AIRELIABLE_LOG_DIR            引擎日志目录（SDK 推导 log_dir 默认值）
    AIRELIABLE_DEVICE_POLICY      设备策略（auto/gpu/gpu+cpu/cpu，平台注入，v0.3.5+）
    STRATA_BINS_DIR               strata-bins 二进制目录
    STRATA_DATA_DIR               Strata 运行时数据目录（与模型目录解耦）
    STRATA_INTERNAL_PORT_START    隐藏端口范围起始（默认 8100）
    STRATA_INTERNAL_PORT_END      隐藏端口范围结束（默认 8200）
    STRATA_SPEC                   MTP 投机解码深度（默认 4，设 0 关闭以隔离问题，R19）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aireliable_engine import EngineConfig as BaseEngineConfig

logger = logging.getLogger(__name__)

# ===== 常量 =====

ENGINE_ID = "strata_llm_7c4f2b91"
ENGINE_NAME = "Strata"
CAPABILITY = "chat"
ENGINE_TYPE = "llm"

# 默认端口（与 config/engines/strata_llm_7c4f2b91.yaml 一致）
DEFAULT_PORT = 7010

# 模型目录布局（相对 MODELS_ROOT）
# 优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE（平台注入）
_DEFAULT_MODEL_SUBDIR = "llm/checkpoints/strata/"

# strata-bins 二进制目录默认值（外部路径，由注册片段 env 注入）
DEFAULT_STRATA_BINS_DIR = "D:/myProInstallDir/VLLM/strata/strata-bins"

# Strata 运行时数据目录默认值（与模型目录解耦；避免 Strata 二次下载 70GB）
DEFAULT_STRATA_DATA_DIR = "D:/myProInstallDir/VLLM/strata/data"

# Strata 源码目录默认值（setup.py / serve/server.py 所在目录）
# 真实 HTTP 服务与打包器都在源码里，strata.exe 只是被 serve 派生的底层引擎。
DEFAULT_STRATA_SRC_DIR = "D:/myProInstallDir/VLLM/strata/_src/Strata-main"

# 本地预构建引擎包目录（setup.py 的 --prebuilt，内含 strata-windows-x64-cuda12.zip）
DEFAULT_STRATA_PREBUILT_DIR = "D:/myProInstallDir/VLLM/strata"

# 引擎 CUDA 工具包（12 = 兼容 Turing/Pascal/Volta 的实验引擎；13 = 默认）
# 本机 Quadro RTX 8000 为 sm_75（Turing），官方 cuda12 zip 的 archs 覆盖 75。
DEFAULT_STRATA_TOOLKIT = 12

# 源码准备（setup.py 装引擎 / 建 pack / 建 tokenizer）超时（秒）：长任务
DEFAULT_PREPARE_TIMEOUT = 3600.0

# 隐藏端口范围（Strata 内部监听，仅 127.0.0.1）
DEFAULT_INTERNAL_PORT_START = 8100
DEFAULT_INTERNAL_PORT_END = 8200

# MTP 投机解码深度（R19：默认 4；可用 STRATA_SPEC 覆盖为 0 以隔离问题）
DEFAULT_SPEC = 4

# 请求超时（秒）：125B 生成是长任务，留足余量
DEFAULT_REQUEST_TIMEOUT = 3600.0
# 模型/子进程启动探针超时（秒）
DEFAULT_LOAD_TIMEOUT = 600.0
# 健康检查轮询间隔（秒）
DEFAULT_HEALTH_CHECK_INTERVAL = 3.0

# 设备策略合法取值（与平台 AIRELIABLE_DEVICE_POLICY 语义对齐，v0.3.5+）
# auto / gpu / gpu+cpu / cpu —— 与 core EngineManager._resolve_device_policy 一致
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量，双源真相消除）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量（模块加载时根据环境变量解析）
MODEL_SUBDIR = _resolve_model_subdir()


@dataclass
class EngineConfig(BaseEngineConfig):
    """Strata LLM 引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)          # GGUF 模型目录（llm/checkpoints/strata/）
    log_dir: Path = field(init=False)            # 引擎 Python 进程日志目录
    server_log_dir: Path = field(init=False)     # strata.exe 子进程日志目录

    # ===== 外部二进制 / 运行时数据 =====
    bins_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("STRATA_BINS_DIR") or DEFAULT_STRATA_BINS_DIR
    ))
    strata_data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("STRATA_DATA_DIR") or DEFAULT_STRATA_DATA_DIR
    ))
    # Strata 源码目录（setup.py / serve/server.py）
    strata_src_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("STRATA_SRC_DIR") or DEFAULT_STRATA_SRC_DIR
    ))
    # 本地预构建引擎包目录（--prebuilt）
    strata_prebuilt_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("STRATA_PREBUILT_DIR") or DEFAULT_STRATA_PREBUILT_DIR
    ))
    # 引擎 CUDA 工具包（12 / 13）
    strata_toolkit: int = int(os.environ.get("STRATA_TOOLKIT") or DEFAULT_STRATA_TOOLKIT)

    # ===== 隐藏端口范围 =====
    internal_port_start: int = DEFAULT_INTERNAL_PORT_START
    internal_port_end: int = DEFAULT_INTERNAL_PORT_END
    # 固定内部端口（优先使用；若设置则忽略端口范围）
    internal_port: Optional[int] = None

    # ===== 子进程 / 请求 =====
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    load_timeout: float = DEFAULT_LOAD_TIMEOUT
    prepare_timeout: float = DEFAULT_PREPARE_TIMEOUT
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL
    spec: int = DEFAULT_SPEC                     # MTP 投机解码深度（0 = 关闭）

    # ===== 默认 variant（模式）=====
    # 优先级：manifest.models.default_variant > 此处 config.default_variant > None
    default_variant: Optional[str] = None

    def __post_init__(self) -> None:
        """推导子路径、统一绝对化并创建运行时目录。

        ⚠️ 必须绝对化（与 video_up_enlarge 同因）：
            本引擎会以 ``cwd=<strata.exe 所在目录>`` 语义调用外部命令，
            子进程收到的 ``--config`` / ``--gguf-dir`` / ``--data-dir`` 等
            路径若为相对路径，会被解析到 **二进制目录** 而非项目根，
            导致模型找不到 / 数据落到错误的盘。在配置层一次性绝对化，
            从根上消除这一整类 cwd 依赖缺陷。
        """
        # 统一绝对化（对已是绝对路径的输入，resolve() 是幂等的）
        self.project_root = Path(self.project_root).resolve()
        self.models_root = Path(self.models_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        self.bins_dir = Path(self.bins_dir).resolve()
        self.strata_data_dir = Path(self.strata_data_dir).resolve()
        self.strata_src_dir = Path(self.strata_src_dir).resolve()
        self.strata_prebuilt_dir = Path(self.strata_prebuilt_dir).resolve()

        # 模型目录（llm/checkpoints/strata/）
        self.model_dir = (self.models_root / MODEL_SUBDIR).resolve()

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = (self.project_root / "logs" / "engines").resolve()
        # strata.exe 子进程日志单独分目录（避免与引擎 Python 日志混淆）
        self.server_log_dir = (
            self.project_root / "logs" / "engines" / "strata_server"
        ).resolve()

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.server_log_dir.mkdir(parents=True, exist_ok=True)
        # 数据目录存在才能作为 strata.exe 的 --data-dir
        self.strata_data_dir.mkdir(parents=True, exist_ok=True)

        # 设备策略归一化（非法值回退 auto）
        if self.device not in DEVICE_POLICIES:
            self.device = "auto"

        # 平台 core URL（若未显式传入，从环境变量读取）
        if self.core_url is None:
            self.core_url = os.environ.get("AIRELIABLE_CORE_URL")

    # ===== 外部二进制路径 =====

    @property
    def strata_exe(self) -> Path:
        """Strata 主可执行文件路径。

        Windows: strata-bins/strata.exe
        Linux/Mac: strata-bins/strata
        """
        name = "strata.exe" if sys.platform == "win32" else "strata"
        return self.bins_dir / name

    @property
    def strata_vision_exe(self) -> Path:
        """Strata 多模态（vision）可执行文件路径。

        Windows: strata-bins/strata-vision.exe
        Linux/Mac: strata-bins/strata-vision
        """
        name = "strata-vision.exe" if sys.platform == "win32" else "strata-vision"
        return self.bins_dir / name

    # ===== Strata 源码脚本路径（真实 HTTP 服务 + 打包器）=====

    @property
    def setup_script(self) -> Path:
        """Strata 打包/装引擎脚本（setup.py）。"""
        return self.strata_src_dir / "setup.py"

    @property
    def server_script(self) -> Path:
        """Strata 真实 HTTP 服务脚本（serve/server.py）。

        ``strata.exe`` 本身不认 ``--config/--port``；真正的 OpenAI 兼容服务
        由 ``serve/server.py`` 提供，它再派生 ``strata.exe --serve <真实参数>``。
        """
        return self.strata_src_dir / "serve" / "server.py"

    def installed_engine_exe(self) -> Path:
        """setup.py 安装的引擎可执行文件（engine-cuda12/ 或 engine/）。"""
        folder = "engine-cuda12" if int(self.strata_toolkit) == 12 else "engine"
        name = "strata.exe" if sys.platform == "win32" else "strata"
        return self.strata_src_dir / folder / name

    def model_config_json(self, tag: str) -> Path:
        """setup.py 为某量化档写出的模型配置 json（``strata-<tag>.json``）。"""
        return self.strata_src_dir / f"strata-{tag.lower()}.json"

    def model_config_log(self, tag: str) -> Path:
        """该量化档的引擎日志（``strata-<tag>.log``）。"""
        return self.strata_src_dir / f"strata-{tag.lower()}.log"

    @property
    def strata_venv_dir(self) -> Path:
        """Strata 官方专用虚拟环境目录（``<src>/.venv``）。

        与平台引擎 env（``envs/llm-strata``）解耦：setup.py 会把自己的
        依赖（numpy/cmake/ninja/… pinned 版本）pip 安装进这里，
        避免污染平台托管的环境。
        """
        return self.strata_src_dir / ".venv"

    @property
    def strata_venv_python(self) -> Path:
        """Strata 专用 venv 的 Python 解释器路径。"""
        if sys.platform == "win32":
            return self.strata_venv_dir / "Scripts" / "python.exe"
        return self.strata_venv_dir / "bin" / "python"

    @property
    def strata_python(self) -> str:
        """用于运行 Strata ``setup.py`` / ``serve/server.py`` 的 Python 解释器。

        解析优先级：
            1. 环境变量 ``STRATA_PYTHON``（显式覆盖）
            2. Strata 专用 venv ``<src>/.venv``（官方 START-HERE 流程；存在即用）
            3. 当前引擎进程解释器（兜底；注意 uv venv 无 pip，setup 会失败）
        """
        override = os.environ.get("STRATA_PYTHON")
        if override:
            return override
        if self.strata_venv_python.exists():
            return str(self.strata_venv_python)
        return sys.executable

    # ===== 设备解析 =====

    def resolve_device(self) -> str:
        """解析推理设备为终态语义（"gpu" / "cpu"）。

        设备策略取值（与平台 ``AIRELIABLE_DEVICE_POLICY`` 语义对齐）：
            auto    → 探测：torch.cuda 可用 → gpu；否则 nvidia-smi 可用 → gpu；否则 cpu
            gpu     → gpu
            gpu+cpu → gpu（显存不足时的层溢出由 strata.exe 自身
                      ``--expert-cache auto`` 机制处理，Python 层不介入）
            cpu     → cpu

        ⚠️ 本引擎不依赖 torch（薄包装层常态），因此 torch 探测失败后**必须**再用
        nvidia-smi 兜底：否则在无 torch 的 GPU 机器上 ``auto`` 会恒回退 cpu，
        导致 strata.exe 走 CPU 后端（125B 模型不可用）。同时 ``auto`` 不再返回
        "auto" 未决态——strata.exe 的 json ``cuda``/``backend`` 字段需要具体布尔值。
        """
        device = (self.device or "auto").strip()
        if device == "cpu":
            return "cpu"
        if device in ("gpu", "gpu+cpu"):
            return "gpu"
        # auto：torch 探测（若恰好可用，优先信任）
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
        # nvidia-smi 兜底（本引擎环境无 torch 的常态）
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8,
                creationflags=(0x08000000 if sys.platform == "win32" else 0),
            )
            if out.returncode == 0 and out.stdout.strip():
                logger.info(
                    "auto 设备解析：torch 不可用，nvidia-smi 探测到 GPU → gpu（%s）",
                    out.stdout.strip().splitlines()[0],
                )
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "server_log_dir": str(self.server_log_dir),
            "bins_dir": str(self.bins_dir),
            "strata_data_dir": str(self.strata_data_dir),
            "strata_src_dir": str(self.strata_src_dir),
            "strata_prebuilt_dir": str(self.strata_prebuilt_dir),
            "strata_toolkit": int(self.strata_toolkit),
            "device": self.device,
            "spec": self.spec,
            "log_level": self.log_level,
        }


def parse_args(argv: Optional[list[str]] = None) -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行参数 > 环境变量 > 默认值。
    """
    parser = argparse.ArgumentParser(description="Strata LLM 推理引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备策略：auto / gpu / gpu+cpu / cpu"
                             "（默认读环境变量 AIRELIABLE_DEVICE_POLICY，再回退 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument("--output-root", type=str, default=None,
                        help="产物根目录（默认 output/）")
    parser.add_argument("--bins-dir", type=str, default=None,
                        help=f"strata-bins 目录（默认 {DEFAULT_STRATA_BINS_DIR}）")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Strata 运行时数据目录（默认 {DEFAULT_STRATA_DATA_DIR}）")
    parser.add_argument("--src-dir", type=str, default=None,
                        help=f"Strata 源码目录（setup.py / serve/server.py，默认 {DEFAULT_STRATA_SRC_DIR}）")
    parser.add_argument("--prebuilt-dir", type=str, default=None,
                        help=f"本地预构建引擎包目录（--prebuilt，默认 {DEFAULT_STRATA_PREBUILT_DIR}）")
    parser.add_argument("--toolkit", type=int, default=None,
                        help=f"引擎 CUDA 工具包（12 / 13，默认 {DEFAULT_STRATA_TOOLKIT}）")
    parser.add_argument("--spec", type=int, default=None,
                        help=f"MTP 投机解码深度（默认 {DEFAULT_SPEC}；0 = 关闭）")
    parser.add_argument("--internal-port-start", type=int, default=None,
                        help=f"隐藏端口范围起始（默认 {DEFAULT_INTERNAL_PORT_START}）")
    parser.add_argument("--internal-port-end", type=int, default=None,
                        help=f"隐藏端口范围结束（默认 {DEFAULT_INTERNAL_PORT_END}）")
    parser.add_argument("--default-variant", type=str, default=None,
                        help="默认模式 variant（默认读 manifest）")

    args = parser.parse_args(argv)

    models_root = Path(
        args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models")
    )
    output_root = Path(
        args.output_root or os.environ.get("AIRELIABLE_OUTPUT_ROOT", "output")
    )
    bins_dir = Path(
        args.bins_dir or os.environ.get("STRATA_BINS_DIR") or DEFAULT_STRATA_BINS_DIR
    )
    strata_data_dir = Path(
        args.data_dir or os.environ.get("STRATA_DATA_DIR") or DEFAULT_STRATA_DATA_DIR
    )
    strata_src_dir = Path(
        args.src_dir or os.environ.get("STRATA_SRC_DIR") or DEFAULT_STRATA_SRC_DIR
    )
    strata_prebuilt_dir = Path(
        args.prebuilt_dir or os.environ.get("STRATA_PREBUILT_DIR") or DEFAULT_STRATA_PREBUILT_DIR
    )
    toolkit = args.toolkit
    if toolkit is None:
        toolkit = int(os.environ.get("STRATA_TOOLKIT") or DEFAULT_STRATA_TOOLKIT)

    start = args.internal_port_start
    if start is None:
        start = int(os.environ.get("STRATA_INTERNAL_PORT_START", DEFAULT_INTERNAL_PORT_START))
    end = args.internal_port_end
    if end is None:
        end = int(os.environ.get("STRATA_INTERNAL_PORT_END", DEFAULT_INTERNAL_PORT_END))

    spec = args.spec
    if spec is None:
        spec = int(os.environ.get("STRATA_SPEC", DEFAULT_SPEC))

    # 设备策略：命令行 --device > 环境变量 AIRELIABLE_DEVICE_POLICY（平台注入）> 默认 auto
    device = (
        args.device
        or os.environ.get("AIRELIABLE_DEVICE_POLICY")
        or "auto"
    ).strip() or "auto"

    return EngineConfig(
        port=args.port,
        host=args.host,
        device=device,
        log_level=args.log_level,
        models_root=models_root,
        output_root=output_root,
        bins_dir=bins_dir,
        strata_data_dir=strata_data_dir,
        strata_src_dir=strata_src_dir,
        strata_prebuilt_dir=strata_prebuilt_dir,
        strata_toolkit=toolkit,
        internal_port_start=start,
        internal_port_end=end,
        spec=spec,
        default_variant=args.default_variant,
    )

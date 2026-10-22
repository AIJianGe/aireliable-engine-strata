"""Strata 子进程封装（StrataProcessManager）。

以「外部二进制 + Python 薄包装层」范式托管 ``strata.exe`` 子进程，
实现 project-rules 要求的 8 条生命周期范式：

    1. 隐藏窗口启动（Windows ``CREATE_NO_WINDOW``）
    2. Job Object 绑定（父进程退出时自动回收子进程树）
    3. 停止时 ``taskkill /T /F`` 杀进程树（Windows）
    4. 子进程日志独立目录 + 每次启动以 ``"w"`` 覆写
       （``logs/engines/strata_server/strata_server.log``）
    5. stdout / stderr 合并重定向到同一日志文件（``stderr=subprocess.STDOUT``）
    6. 内部端口 8100–8200 分配与占用检测（仅 127.0.0.1）
    7. ``wait_ready()`` 轮询子进程 HTTP 端点确认就绪
    8. 退出时记录 exit code + 最近 N 行日志

两段式启动（返工后：贴合 Strata 官方真实契约）：

    ⚠️ ``strata.exe`` **本身不认** ``--engine/--config/--port/--gguf-dir``；
    真正的 OpenAI 兼容 HTTP 服务是 Strata 源码里的 ``serve/server.py``，
    它内部再派生 ``strata.exe --serve <真实参数>``。真实运行参数由
    ``setup.py`` 写进 ``strata-<tag>.json``。

    - A 段 ``_run_setup``：调用官方 ``setup.py --no-start``（幂等）
        · 装引擎（本地预构建 zip → ``<src>/engine-cuda12/strata.exe``）
        · 建 pack / tokenizer（``tools/iq_pack.py``，索引而非 68GB 拷贝）
        · 写配置 json（``<src>/strata-<tag>.json``）
        · 用 Strata 专用 venv（``<src>/.venv``）运行，避免污染平台 env
    - ``_patch_model_config``：把本次的 ctx / port / spec 落进配置 json（原子写）
    - B 段 ``start``：``python serve/server.py --engine strata --config <json> --port <p>``
    - ``switch_mode(old, new)``：**停 → 改写 → 冷启动** 三步编排 + 失败回滚
    - ``_switch_lock`` 单飞：同一时刻只允许一个模式切换在途

⚠️ 与 llama_cpp_9cdb76f8 的差异：
    - llama 用 router mode 一次托管多个 GGUF；Strata 每个模式需独立冷启动
    - Strata 的 ``--data-dir`` 必须与平台模型目录解耦，
      否则 Strata 首次启动会试图二次下载 ~70GB 权重（T5.5 规避点）
    - ``--expert-cache`` 由 setup.py 一律锁 ``auto``（绝不写死窗口大小，R17）
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import EngineConfig

logger = logging.getLogger(__name__)

# ===== 常量 =====

# 子进程日志文件名
_SERVER_LOG_NAME = "strata_server.log"


# ===== Windows 进程控制辅助 =====


def _get_creation_flags() -> int:
    """返回子进程创建标志。

    Windows: ``CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP``
        - CREATE_NO_WINDOW (0x08000000)：隐藏控制台窗口
        - CREATE_NEW_PROCESS_GROUP (0x00000200)：便于向进程组发信号
    其他平台: 0
    """
    if sys.platform == "win32":
        return 0x08000000 | 0x00000200
    return 0


def _kill_process_tree(pid: int) -> None:
    """杀死进程树（Windows 用 taskkill /T /F，其他平台用 killpg）。"""
    if pid <= 0:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_get_creation_flags(),
                timeout=15,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("taskkill 杀进程树失败（pid=%s）: %s", pid, e)
        return
    # POSIX
    try:
        import signal

        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception as e:  # noqa: BLE001
        logger.warning("killpg 杀进程树失败（pid=%s）: %s", pid, e)


# ----- Job Object（父进程退出时自动回收子进程树）-----

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _assign_to_job_object(pid: int) -> Optional[int]:
    """把进程加入一个 ``KILL_ON_JOB_CLOSE`` 的 Job Object。

    父进程（引擎 Python 进程）异常退出时，OS 会自动回收该 Job 下所有子进程，
    避免 strata.exe 变成孤儿进程占用显存。

    Returns:
        Job Object 句柄（调用方负责在清理阶段 CloseHandle）；失败返回 None
    """
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            logger.warning("CreateJobObjectW 失败（err=%s）", ctypes.get_last_error())
            return None

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            logger.warning("SetInformationJobObject 失败（err=%s）", ctypes.get_last_error())
            kernel32.CloseHandle(job)
            return None

        PROCESS_TERMINATE = 0x0001
        PROCESS_SET_QUOTA = 0x0100
        proc = kernel32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, pid)
        if not proc:
            logger.warning("OpenProcess 失败（err=%s）", ctypes.get_last_error())
            kernel32.CloseHandle(job)
            return None

        ok = kernel32.AssignProcessToJobObject(job, proc)
        kernel32.CloseHandle(proc)
        if not ok:
            logger.warning("AssignProcessToJobObject 失败（err=%s）", ctypes.get_last_error())
            kernel32.CloseHandle(job)
            return None

        return job
    except Exception as e:  # noqa: BLE001
        logger.warning("Job Object 绑定异常: %s", e)
        return None


def _close_job_object(job: Optional[int]) -> None:
    """关闭 Job Object 句柄（触发 KILL_ON_JOB_CLOSE）。"""
    if not job or sys.platform != "win32":
        return
    try:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
    except Exception:  # noqa: BLE001
        pass


# ===== 端口分配 =====


def _port_in_use(port: int) -> bool:
    """检测端口是否被占用（仅 127.0.0.1）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def allocate_internal_port(start: int = 8100, end: int = 8200) -> int:
    """在 [start, end] 范围内分配一个空闲端口。

    Raises:
        RuntimeError: 范围内无空闲端口
    """
    for port in range(start, end + 1):
        if not _port_in_use(port):
            return port
    raise RuntimeError(f"隐藏端口范围 {start}-{end} 内无空闲端口")


# ===== 就绪轮询 =====


async def wait_for_ready(
    internal_port: int,
    timeout: float,
    interval: float = 3.0,
    process: Optional[subprocess.Popen] = None,
) -> bool:
    """轮询子进程 HTTP 端点，确认 Strata 就绪。

    依次探测 ``/health`` 与 ``/v1/models``（Strata 的 OpenAI 兼容端点）。

    Returns:
        True 表示就绪；超时返回 False
    """
    deadline = time.time() + timeout
    urls = [
        f"http://127.0.0.1:{internal_port}/health",
        f"http://127.0.0.1:{internal_port}/v1/models",
    ]
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), trust_env=False) as client:
        while time.time() < deadline:
            if process is not None and process.poll() is not None:
                logger.error(
                    "Strata 子进程在就绪前退出（exit code=%s）", process.returncode
                )
                return False
            for url in urls:
                try:
                    resp = await client.get(url)
                    if resp.status_code < 500:
                        logger.info("Strata 已就绪（%s → HTTP %s）", url, resp.status_code)
                        return True
                except Exception:  # noqa: BLE001
                    pass
            await asyncio.sleep(interval)
    logger.error("Strata 就绪轮询超时（%.0fs，端口 %s）", timeout, internal_port)
    return False


# ===== 进程管理器 =====


class StrataProcessManager:
    """托管 strata.exe 子进程 + 模式冷切换编排。"""

    def __init__(self, config: EngineConfig, logger_: Optional[logging.Logger] = None):
        self.config = config
        self._logger = logger_ or logging.getLogger(__name__)

        self._process: Optional[subprocess.Popen] = None
        self._log_file_handle = None
        self._job_handle: Optional[int] = None

        self._internal_port: Optional[int] = None
        self.current_mode: Optional[str] = None
        # 模式名 → 当前生效上下文长度（供 /status 展示）
        self.current_ctx: Optional[int] = None

        # 模式切换单飞锁
        self._switch_lock = asyncio.Lock()

    # ===== 属性 =====

    @property
    def is_running(self) -> bool:
        """子进程是否存活。"""
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        """子进程 PID。"""
        return self._process.pid if self._process else None

    @property
    def internal_port(self) -> Optional[int]:
        """子进程内部监听端口。"""
        return self._internal_port

    @property
    def internal_url(self) -> Optional[str]:
        """子进程 OpenAI 兼容端点基址。"""
        if self._internal_port is None:
            return None
        return f"http://127.0.0.1:{self._internal_port}"

    # ===== 启动 =====

    async def start(
        self,
        mode_name: str,
        ctx: int,
        spec: int,
        vision: bool,
        device: str,
    ) -> None:
        """启动 strata.exe 子进程并等待就绪。

        Args:
            mode_name: 模式 variant 名（如 strata/balanced）
            ctx: 上下文长度
            spec: MTP 投机解码深度（0 = 关闭）
            vision: 是否启用多模态
            device: 已解析的设备（"gpu" / "cpu"）

        Raises:
            RuntimeError: 可执行文件缺失 / 端口耗尽 / 就绪超时
        """
        if self.is_running:
            self._logger.info("Strata 子进程已在运行（pid=%s），先停止再启动", self.pid)
            await self.stop()

        # 0) 源码/脚本存在性校验
        self._validate_source()

        # 1) 分配内部端口
        if self.config.internal_port is not None:
            self._internal_port = int(self.config.internal_port)
        else:
            self._internal_port = allocate_internal_port(
                self.config.internal_port_start, self.config.internal_port_end
            )

        # 2) 模式 → Quant 档（决定 setup.py 的 --family/--model 与配置 json 名）
        family, model, tag, mode_mtp, mode_vision = self._mode_to_model(mode_name)
        vision = bool(vision or mode_vision)

        # 3) A 段（prepare）：官方 setup.py 装引擎 + 建 pack/tokenizer + 写配置 json（幂等）
        cfg_json = self.config.model_config_json(tag)
        if not cfg_json.exists():
            self._logger.info(
                "A 段 prepare：配置 json 不存在，调用官方 setup.py（family=%s, model=%s, ctx=%s）",
                family, model, ctx,
            )
            await asyncio.to_thread(
                self._run_setup, family, model, ctx, vision, self._internal_port
            )
        else:
            self._logger.info("A 段 prepare：复用已有配置 json %s", cfg_json)

        # 4) 把本次请求的 ctx / port / spec 落到配置 json（引擎真实参数在 json 的 args 里）
        self._patch_model_config(cfg_json, ctx=ctx, port=self._internal_port, spec=spec)

        # 5) B 段（serve）：启动真实 HTTP 服务 serve/server.py（内部派生 strata.exe）
        cmd = [
            self._ensure_strata_python(),
            str(self.config.server_script),
            "--engine", "strata",
            "--config", str(cfg_json),
            "--port", str(self._internal_port),
        ]

        # 6) 打开子进程日志（"w" 覆写）
        self.config.server_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.config.server_log_dir / _SERVER_LOG_NAME
        self._log_file_handle = open(log_file, "w", encoding="utf-8")

        self._logger.info(
            "启动 Strata 服务：cmd=%s | cwd=%s | 内部端口=%s | 模式=%s | ctx=%s | spec=%s | vision=%s",
            " ".join(cmd), self.config.strata_src_dir, self._internal_port,
            mode_name, ctx, spec, vision,
        )

        # 7) Popen（隐藏窗口 + 合并输出）
        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.config.strata_src_dir),
            stdout=self._log_file_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_get_creation_flags(),
            start_new_session=(sys.platform != "win32"),
            close_fds=True,
        )

        # 8) Job Object 绑定
        self._job_handle = _assign_to_job_object(self._process.pid)

        # 9) 就绪轮询
        ready = await wait_for_ready(
            internal_port=self._internal_port,
            timeout=self.config.load_timeout,
            interval=self.config.health_check_interval,
            process=self._process,
        )
        if not ready:
            recent = self.read_recent_logs(40)
            await self.stop()
            raise RuntimeError(
                f"Strata 子进程启动失败或就绪超时（端口 {self._internal_port}）。\n"
                f"最近日志：\n{recent}"
            )

        self.current_mode = mode_name
        self.current_ctx = ctx
        self._logger.info(
            "Strata 启动完成：pid=%s，内部端口=%s，模式=%s", self.pid, self._internal_port, mode_name
        )

    # ===== 停止 =====

    async def stop(self) -> None:
        """停止子进程树并清理资源（幂等）。"""
        proc = self._process
        if proc is not None:
            pid = proc.pid
            rc = proc.poll()
            if rc is None:
                self._logger.info("停止 Strata 子进程树：pid=%s", pid)
                _kill_process_tree(pid)
                try:
                    proc.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    try:
                        proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:  # noqa: BLE001
                        try:
                            proc.kill()
                        except Exception:  # noqa: BLE001
                            pass
                rc = proc.poll()
            recent = self.read_recent_logs(20)
            self._logger.info(
                "Strata 子进程已停止：pid=%s，exit code=%s\n最近日志：\n%s", pid, rc, recent
            )
        self._cleanup_resources()
        self.current_mode = None
        self.current_ctx = None

    def _cleanup_resources(self) -> None:
        """释放文件句柄 / Job Object / 端口记录。"""
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_file_handle = None
        _close_job_object(self._job_handle)
        self._job_handle = None
        self._process = None
        self._internal_port = None

    # ===== 模式编排（阶段 3）=====

    async def switch_mode(
        self, old_mode: Optional[str], new_mode: str, ctx: int, spec: int,
        vision: bool, device: str,
    ) -> None:
        """冷切换模式：**停 → 改写 → 冷启动** 三步编排 + 失败回滚。

        ``_switch_lock`` 保证同一时刻只有一个切换在途（单飞）。

        Raises:
            RuntimeError: 新模式启动失败（此时已尝试回滚到 old_mode）
        """
        async with self._switch_lock:
            if self.current_mode == new_mode and self.is_running:
                self._logger.info("模式已是 %s，无需切换", new_mode)
                return

            previous = old_mode or self.current_mode
            self._logger.info("模式切换开始：%s → %s", previous, new_mode)

            # 步骤 1+2+3：停 → 改写（在 start 内部完成）→ 冷启动
            try:
                await self.start(
                    mode_name=new_mode, ctx=ctx, spec=spec, vision=vision, device=device
                )
            except Exception as e:  # noqa: BLE001
                self._logger.error("新模式 %s 启动失败：%s，尝试回滚到 %s", new_mode, e, previous)
                if previous and previous != new_mode:
                    try:
                        await self._rollback_mode(previous, device)
                    except Exception as re_:  # noqa: BLE001
                        self._logger.error("回滚失败：%s", re_)
                        raise RuntimeError(
                            f"切换模式到 {new_mode} 失败，且回滚到 {previous} 也失败：{e} / {re_}"
                        ) from e
                raise

    async def _rollback_mode(self, old_mode: str, device: str) -> None:
        """回滚到旧模式（用旧模式的默认参数冷启动）。

        旧模式的 ctx/spec/vision 从 MODES 定义反查（避免依赖调用方传入）。
        """
        from .schemas import get_mode

        mode_def = get_mode(old_mode)
        if not mode_def:
            raise RuntimeError(f"无法回滚：未知模式 {old_mode}")
        await self.start(
            mode_name=old_mode,
            ctx=int(mode_def.get("ctx", 131072)),
            spec=self.config.spec if mode_def.get("mtp") else 0,
            vision=bool(mode_def.get("vision", False)),
            device=device,
        )

    # ===== A 段：源码准备（官方 setup.py 装引擎 + 打包 + 写配置）=====

    def _validate_source(self) -> None:
        """校验 Strata 源码脚本（setup.py / serve/server.py）存在。"""
        missing: list[str] = []
        if not self.config.setup_script.exists():
            missing.append(str(self.config.setup_script))
        if not self.config.server_script.exists():
            missing.append(str(self.config.server_script))
        if missing:
            raise RuntimeError(
                "Strata 源码脚本缺失（真实 HTTP 服务与打包器都在源码里）：\n  "
                + "\n  ".join(missing)
                + f"\n  · 期望源码目录：{self.config.strata_src_dir}\n"
                "  · 请到引擎详情页 → 模型管理 → 下载 Strata 源码，"
                "或手动放置后重试（官方仓库：https://github.com/Niko1221/Strata）"
            )

    @staticmethod
    def _mode_to_model(mode_name: str) -> tuple[str, str, str, bool, bool]:
        """模式名 → (family, model, tag, mtp, vision)。

        - family : setup.py 的 ``--family``（main→qwen，coder→coder）
        - model  : setup.py 的 ``--model``（量化档键，如 IQ2_XS）
        - tag    : setup.py 的标签（``fam["tag"] + model``）；配置 json = ``strata-<tag>.json``
        """
        from .schemas import get_mode

        md = get_mode(mode_name)
        if not md:
            raise RuntimeError(f"未知模式（schemas.MODES 中不存在）: {mode_name}")
        family = "coder" if md.get("family") == "coder" else "qwen"
        model = str(md.get("quant_key") or "").strip()
        if not model:
            raise RuntimeError(f"模式 {mode_name} 未声明 quant_key，无法映射 Strata 量化档")
        prefix = "coder-" if family == "coder" else ""
        return family, model, f"{prefix}{model}", bool(md.get("mtp")), bool(md.get("vision"))

    def _ensure_strata_python(self) -> str:
        """返回可运行 Strata 脚本的 Python，必要时先创建专用 venv。

        官方 ``setup.py`` 会用 ``sys.executable -m pip install`` 装自己的
        依赖；平台引擎 env（uv venv）**不带 pip**，因此必须用 Strata 自己的
        ``<src>/.venv``（官方 START-HERE 流程）来跑。
        """
        override = os.environ.get("STRATA_PYTHON")
        if override:
            return override
        venv_py = self.config.strata_venv_python
        if venv_py.exists():
            return str(venv_py)
        self._create_strata_venv(venv_py)
        return str(venv_py)

    def _create_strata_venv(self, venv_py: Path) -> None:
        """用系统 Python 创建 Strata 专用 venv（``<src>/.venv``）。"""
        base = self._find_bootstrap_python()
        self._logger.info(
            "创建 Strata 专用 venv：%s（base python=%s）", self.config.strata_venv_dir, base
        )
        proc = subprocess.run(
            [base, "-m", "venv", str(self.config.strata_venv_dir)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=_get_creation_flags(),
        )
        if proc.returncode != 0 or not venv_py.exists():
            tail = "\n".join(
                ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-30:]
            )
            raise RuntimeError(
                f"创建 Strata 专用 venv 失败（base={base}，exit code={proc.returncode}）：\n{tail}"
            )

    @staticmethod
    def _find_bootstrap_python() -> str:
        """定位一个带 pip/venv 的系统 Python（优先官方 START-HERE 的查找顺序）。"""
        import shutil as _sh

        # 1) PATH 上的 python（本机已验证为 3.12 + pip）
        p = _sh.which("python")
        if p:
            return p
        # 2) py 启动器
        py = _sh.which("py")
        if py:
            return py
        # 3) Windows 常见的按用户安装路径
        local = os.environ.get("LOCALAPPDATA")
        if local:
            for ver in ("313", "312", "311", "310"):
                cand = Path(local) / "Programs" / "Python" / f"Python{ver}" / "python.exe"
                if cand.exists():
                    return str(cand)
        raise RuntimeError(
            "找不到可用于创建 Strata venv 的 Python（需 3.10+ 且带 pip）："
            "请安装 Python 3.12 或设置 STRATA_PYTHON 指向已有的 Python 解释器"
        )

    def _pip_index_url(self) -> str:
        """解析 pip 镜像源（供 setup.py 的 pip install 使用）。

        优先级：``PIP_INDEX_URL`` 环境变量 > ``UV_DEFAULT_INDEX`` 环境变量 >
        平台配置 ``<project_root>/config/uv_mirrors.yaml`` 的 ``uv_mirrors.current``。
        为空表示用官方源。
        """
        for key in ("PIP_INDEX_URL", "UV_DEFAULT_INDEX"):
            v = os.environ.get(key)
            if v:
                return v.strip()
        cfg = self.config.project_root / "config" / "uv_mirrors.yaml"
        if cfg.exists():
            try:
                import yaml

                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                cur = (data.get("uv_mirrors") or {}).get("current") or ""
                return str(cur).strip()
            except Exception as e:  # noqa: BLE001
                self._logger.debug("读取 uv_mirrors.yaml 失败：%s", e)
        return ""

    def _run_setup(self, family: str, model: str, ctx: int, vision: bool, port: int) -> None:
        """调用 Strata 官方 setup.py 完成安装与打包（``--no-start``，幂等）。

        这一步产出：
            - ``<src>/engine-cuda12/strata.exe``（本地 zip 安装，不联网下载二进制）
            - ``<data>/packs/<tag>/``（iq_pack 索引 + tokenizer）
            - ``<src>/strata-<tag>.json``（引擎真实运行配置）

        Raises:
            RuntimeError: setup 失败/超时（含最近输出）
        """
        cmd = [
            self._ensure_strata_python(),
            str(self.config.setup_script),
            "--family", family,
            "--model", model,
            "--context", str(int(ctx)),
            "--kv", "int8",
            "--vision", "yes" if vision else "no",
            "--data-dir", str(self.config.strata_data_dir),
            "--gguf-dir", str(self.config.model_dir),
            "--cuda", str(int(self.config.strata_toolkit)),
            "--prebuilt", str(self.config.strata_prebuilt_dir),
            "--port", str(int(port)),
            "--host", "127.0.0.1",
            "--yes",
            "--no-start",
            "--no-browser",
        ]
        env = dict(os.environ)
        # MTP 抓取走镜像（huggingface.co 直连不可达，仅 hf-mirror 可用）
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # setup.py 会用自己的 venv 跑 pip install 装依赖；国内直连 PyPI 官方源不稳，
        # 复用平台镜像配置（config/uv_mirrors.yaml）注入 PIP_INDEX_URL
        idx = self._pip_index_url()
        if idx:
            env["PIP_INDEX_URL"] = idx
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            self._logger.info("Strata setup.py 使用 pip 镜像源：%s", idx)

        self._logger.info("调用 Strata setup.py（可能耗时数分钟）：%s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.config.strata_src_dir), env=env,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self.config.prepare_timeout,
                creationflags=_get_creation_flags(),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Strata setup.py 超时（{self.config.prepare_timeout:.0f}s）：{e}"
            ) from e

        tail = "\n".join(
            ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-60:]
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Strata setup.py 失败（exit code={proc.returncode}）\n最近输出：\n{tail}"
            )
        self._logger.info("Strata setup.py 完成（exit code=0）\n最近输出：\n%s", tail)

    def _patch_model_config(self, cfg_json: Path, ctx: int, port: int, spec: int) -> None:
        """把本次的 ctx / port / spec 落进 setup.py 生成的配置 json（原子写）。

        - ``port``：serve 监听端口
        - ``args`` 里的 ``--max-context``：上下文长度
        - ``spec`` > 0：写入 MTP 投机深度
        - ``spec`` == 0：只移除 ``--mtp``（关闭 MTP 草稿头），**保留** ``--spec``/``--spec-min-p``

        ⚠️ 为什么 spec==0 不能删 ``--spec``：
            ``--spec`` 与 ``--mtp`` 是两件事。``--mtp`` 是独立的 MTP 草稿模型；
            而 ``--spec T (T>=2)`` 是「原生 IQ pack」多 token 解码内核的**硬性要求**
            （strata.exe 实测：删掉 ``--spec`` 直接 exit code 2，报
            ``... is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK``）。
            setup.py 默认一律写 ``--spec 4 --spec-min-p 0.5``，本函数不应对其做减法。

        基线（baseline）：
            多个模式可能**共用同一个 json**（``strata/balanced``、``strata/long``、
            ``strata/compact`` 都是 IQ2_XS → 同一个 ``strata-iq2_xs.json``）。
            若直接在（已被上一次 patch 改过的）json 上继续增删，模式之间会互相污染
            （实测：long 删掉 ``--mtp`` 后，balanced 再启动就再也拿不回 MTP）。
            因此首次调用时把 setup.py 原始产物留一份 ``<json>.baseline``，
            之后**每次都从基线重新 patch**，保证任意模式反复冷切换都是幂等的。
        """
        base = cfg_json.with_name(cfg_json.name + ".baseline")
        if not base.exists():
            shutil.copy2(cfg_json, base)
            self._logger.info("已留存配置基线：%s", base.name)
        data = json.loads(base.read_text(encoding="utf-8-sig"))
        data["port"] = int(port)
        args: list[str] = list(data.get("args") or [])

        if "--max-context" in args[:-1]:
            args[args.index("--max-context") + 1] = str(int(ctx))
        else:
            args += ["--max-context", str(int(ctx))]

        def _drop(flag: str) -> None:
            while flag in args:
                i = args.index(flag)
                del args[i:i + 2]

        if spec and spec > 0:
            if "--spec" in args[:-1]:
                args[args.index("--spec") + 1] = str(int(spec))
            else:
                args += ["--spec", str(int(spec))]
        else:
            # spec == 0：关闭 MTP 草稿头，但保留 --spec / --spec-min-p
            # （原生 IQ pack 要求 --spec T >= 2，删掉会导致 strata.exe exit code 2）
            _drop("--mtp")
            # 兜底：万一 json 里没有 --spec（或值非法），补一个满足 IQ pack 下限的值
            cur = None
            if "--spec" in args[:-1]:
                try:
                    cur = int(args[args.index("--spec") + 1])
                except ValueError:
                    cur = None
            if cur is None or cur < 2:
                if "--spec" in args[:-1]:
                    args[args.index("--spec") + 1] = "2"
                else:
                    args += ["--spec", "2"]

        data["args"] = args
        tmp = cfg_json.with_suffix(cfg_json.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, cfg_json)
        self._logger.info(
            "已落配置：%s（port=%s, max-context=%s, spec=%s）",
            cfg_json.name, port, ctx, spec,
        )

    # ===== 健康检查 / 日志 =====

    async def health(self) -> bool:
        """探测子进程是否健康。"""
        if not self.is_running or self._internal_port is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), trust_env=False) as client:
                for path in ("/health", "/v1/models"):
                    try:
                        resp = await client.get(f"http://127.0.0.1:{self._internal_port}{path}")
                        if resp.status_code < 500:
                            return True
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            pass
        return False

    def read_recent_logs(self, n_lines: int = 40) -> str:
        """读取子进程日志最近 N 行（用于启动失败时回传）。"""
        log_file = self.config.server_log_dir / _SERVER_LOG_NAME
        if not log_file.exists():
            return "(暂无子进程日志)"
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n_lines:])
        except Exception as e:  # noqa: BLE001
            return f"(读取子进程日志失败: {e})"

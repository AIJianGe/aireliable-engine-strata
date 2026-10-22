"""Strata LLM 推理引擎 HTTP 服务（FastAPI）。

实现 8 个标准 HTTP 端点 + 2 个增强端点 + 挂载输出目录 + 原生 WebUI。

启动：
    uv run --no-sync --project envs/llm-strata \\
        python -m engines.strata_llm_7c4f2b91.server --port 7010

标准端点（SDK 契约，全部引擎一致）：
    GET  /health         健康检查
    GET  /capabilities   能力声明（6 个命名模式，含 recommended 标记）
    POST /load           校验 bins + GGUF 分片（**只校验不下载**）→ 冷启动 strata.exe
    POST /unload         停止子进程 + 释放状态
    POST /infer          对话生成（SSE 流式增量推送）
    GET  /status         状态查询（模式 / 内部端口 / pid / 硬件画像）
    POST /lora/load      LoRA 加载（v1 预留，不支持）
    POST /lora/unload    LoRA 卸载（v1 预留，不支持）

增强端点（本引擎特有）：
    GET  /hardware       硬件探针（GPU / 内存 / 磁盘 / CPU 特性）+ 推荐模式
    POST /mode/switch    冷切换模式（停 → 改写 → 冷启动，失败回滚）

静态 / 元信息：
    GET  /output/*       静态产物文件（生成的文本）
    GET  /ui/*           **原生 Strata UI 的透明别名前缀**（见下）
    GET  /info           引擎信息 JSON（恒可用）
    GET  /               307 跳 /ui/（有 UI）；否则返回 /info
    GET|POST /log_level  日志级别查询 / 调整（SDK 注册）

「/ui/ = 原生 UI 别名前缀」设计（v0.2.20+，替换旧自写 WebUI）：
    平台只保留**一套**界面 —— Strata 官方 ``serve/web`` 原生 UI，零拷贝复用
    （原生文件永远从 Strata 源码目录实时读取，仓库内不存副本）。

    - ``/ui/{path}`` **一律反代**当前内部端口（8100–8200）的 ``/{path}``，
      SSE（text/event-stream）逐块透传不缓冲；原生页内的相对路径
      （``web/app.js``、``fetch("health")``）在 ``/ui/`` 前缀下原样成立。
    - 仅两处例外：``/ui/_platform.js``、``/ui/_platform.css`` 由本引擎**本地**提供
      —— 这两份是「平台适配注入层」（模式切换 / 状态 / 硬件 / 中文文案）。
    - ``/ui/`` 根路由反代原生 ``/``（index.html）后**响应时拼接**注入层两个标签，
      并隐藏原生 ``Model settings`` 卡片（其 ``POST /config`` 语义与冷切换冲突）。
    - 引擎未加载（无内部端口）时 ``/ui/`` 返回本地 ``launcher.html`` 启动器兜底页
      （选模式 + ``POST /load`` + 进度），保持「``/load`` 才冷启动」契约不变。
    - 冲突隔离：原生同名端点（``/health`` ``/status`` ``/load``）住 ``/ui/*``，
      平台契约那份住 ``/*``，前缀天然隔离。

⚠️ 关键契约（详见 __init__.py 与 ANALYSIS-TASKS/61-*）：
    1. load() 只校验不下载；启动失败须抛含「缺失清单 + 路径 + 仓库 + 下载指引」的错
    2. 推理失败**必须真实发 stage=error 帧**（R25：见 engine._infer_impl）
    3. --expert-cache 锁 auto（R17）；MTP spec 默认 4（R19）
    4. 音色管理（data/voice_library.db / data/voices/）零触碰
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# ===== sys.path 注入（保证任意 cwd 下均可启动）=====
# 以 `python -m engines.<id>.server` 启动时项目根已在 sys.path；
# 但若被以 `python server.py` 或绝对路径方式拉起，包相对导入会失败。
# 这里显式注入项目根 + 引擎根目录，兼容两种启动方式。
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
_PROJECT_ROOT = _ENGINE_ROOT.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import StrataEngine
from .schemas import MODES

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
_engine: Optional[StrataEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> StrataEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动初始化引擎，关闭清理资源。"""
    global _engine, _config

    if _config is None:
        _config = parse_args()

    # 配置日志（v0.2.15+ 统一到 logs/engines/，依赖平台注入 AIRELIABLE_LOG_DIR）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== Strata LLM 推理引擎启动 =====")
    logger.info("配置: %s", _config.to_dict())

    _engine = StrataEngine(_config)
    # 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s, scan_path=%s",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
        _engine.default_variant,
        _engine.scanner.last_scan_mode,
    )

    yield

    # 关闭清理：停止 strata.exe 子进程
    logger.info("===== Strata LLM 推理引擎关闭 =====")
    if _engine is not None and _engine.process.is_running:
        try:
            await _engine._unload_impl()
        except Exception as e:  # noqa: BLE001
            logger.warning("关闭时停止 Strata 子进程失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="Strata LLM Engine",
        description=(
            "AIReliable 平台的 Strata LLM 推理引擎"
            "（路线 C：外部二进制 strata.exe + 6 命名模式冷切换）"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用 + iframe 嵌入）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册标准 + 增强端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供产物下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    # ===== 挂载 /ui/（原生 UI 别名前缀 + 平台注入层）=====
    webui_mounted = _mount_platform_ui(app)

    # ===== 根路径（依赖是否挂了 WebUI）=====
    _register_root(app, webui_mounted)

    return app


# ===== /ui/ = 原生 Strata UI 的透明别名前缀 =====

# 本引擎自有的「平台适配注入层」文件名（本地提供，不反代）
_PLATFORM_JS = "_platform.js"
_PLATFORM_CSS = "_platform.css"

# 反代时需剥离的 hop-by-hop / 由本层重算的头
_STRIP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-encoding", "host",
}


# ``/ui/`` 的「本地内容」必须逐次校验缓存：引擎冷启动完成后浏览器若命中旧缓存，
# 会继续显示「未加载」启动器页（v0.4.x 真实浏览器实测踩中）。
_NO_CACHE = {"Cache-Control": "no-cache"}


def _set_header(headers: dict[str, str], key: str, value: str) -> None:
    """大小写不敏感地设置响应头（httpx 会把上游头名小写化，直接赋值会产生重复头）。"""
    for existing in [k for k in headers if k.lower() == key.lower()]:
        del headers[existing]
    headers[key] = value


# 注入到原生 index.html ``</body>`` 前的两个标签
_INJECT_TAGS = (
    f'\n<link rel="stylesheet" href="/ui/{_PLATFORM_CSS}">'
    f'\n<script src="/ui/{_PLATFORM_JS}" defer></script>\n'
)


def _native_base_url() -> Optional[str]:
    """返回当前原生服务基址（引擎未加载 / 未运行 → None）。

    模式冷切换后内部端口会变（8100 ↔ 8101），因此**每次请求**都重新解析，
    保证浏览器恒停在 ``7010/ui/`` 也能命中当前实例。
    """
    try:
        engine = get_engine()
    except RuntimeError:
        return None
    if not engine.process.is_running:
        return None
    return engine.process.internal_url


def _mount_platform_ui(app: FastAPI) -> bool:
    """挂载 ``/ui/``（原生 UI 别名前缀 + 平台注入层）。

    仓库内只保留自己写的 3 个文件（``launcher.html`` / ``_platform.js`` /
    ``_platform.css``）；原生 UI 的全部文件都在请求时从 Strata 源码目录反代，
    **零拷贝**，因此原生升级后无需同步。

    Returns:
        True 表示已挂载；False 表示 ``webui/launcher.html`` 缺失（保持 headless）
    """
    ui_dir = Path(__file__).resolve().parent / "webui"
    launcher = ui_dir / "launcher.html"
    if not (ui_dir.is_dir() and launcher.is_file()):
        logger.info("平台 UI 未挂载（webui/launcher.html 不存在），保持 headless")
        return False

    @app.get("/ui", include_in_schema=False)
    async def _ui_bare():
        """``/ui`` → ``/ui/``（相对路径解析依赖尾斜杠）。"""
        return RedirectResponse(url="/ui/", status_code=307)

    # 方法全集：原生 UI 会 POST /v1/chat/completions、/v1/images/generations，
    # 以及 DELETE /settings 之类；反代层必须逐方法透传，不能只挂 GET。
    @app.api_route(
        "/ui/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def _ui(request: Request, path: str):
        """``/ui/{path}``：本地注入层 → 未加载兜底页 → 反代原生。"""
        # 1) 平台注入层（本地提供，绝不反代 —— 原生无同名文件）
        if path == _PLATFORM_JS:
            return FileResponse(
                ui_dir / _PLATFORM_JS,
                media_type="text/javascript; charset=utf-8",
                headers=_NO_CACHE,
            )
        if path == _PLATFORM_CSS:
            return FileResponse(
                ui_dir / _PLATFORM_CSS,
                media_type="text/css; charset=utf-8",
                headers=_NO_CACHE,
            )

        # 2) 引擎未加载 → 启动器兜底页（保持 /load 才冷启动的契约）
        base = _native_base_url()
        if base is None:
            if path in ("", "/"):
                return FileResponse(
                    launcher, media_type="text/html; charset=utf-8", headers=_NO_CACHE
                )
            raise HTTPException(status_code=503, detail="引擎未加载，请先 POST /load")

        # 3) 其余一律反代原生（SSE 流式透传）
        return await _proxy_to_native(request, base, path)

    logger.info("平台 UI 挂载: /ui/* -> 原生 UI 反代（注入层 %s/%s）", _PLATFORM_JS, _PLATFORM_CSS)
    return True


async def _proxy_to_native(request: Request, base: str, path: str) -> Response:
    """把 ``/ui/{path}`` 反代到原生 ``{base}/{path}``。

    - SSE（``text/event-stream``）逐块透传，**不缓冲**（否则聊天流式失效）
    - 原生 ``/``（index.html）响应拼接平台注入层两个标签
    - 其余响应原样透传（剥离 hop-by-hop 头；gzip 由 httpx 自动解压后重算长度）
    """
    target = f"{base}/{path}" if path else f"{base}/"
    if request.url.query:
        target += f"?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS
    }

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        trust_env=False,
    )
    body = await request.body()
    upstream = client.build_request(
        request.method, target, headers=headers, content=body or None
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"原生服务不可达（{target}）：{e}")

    ctype = resp.headers.get("content-type", "")

    # ---- SSE / 流式：逐块透传，不可 aread() ----
    if "text/event-stream" in ctype or "application/x-ndjson" in ctype:
        out_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _STRIP_HEADERS and k.lower() != "cache-control"
        }
        out_headers["Cache-Control"] = "no-cache"
        out_headers["X-Accel-Buffering"] = "no"

        async def _stream():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            _stream(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
            headers=out_headers,
        )

    content = await resp.aread()
    await resp.aclose()
    await client.aclose()

    out_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _STRIP_HEADERS
    }

    # ---- 原生首页：拼接注入层（零拷贝的代价就是这个字符级拼接）----
    if path in ("", "/", "index.html") and "text/html" in ctype:
        html = content.decode("utf-8", errors="replace")
        if _PLATFORM_JS not in html:
            if "</body>" in html:
                html = html.replace("</body>", _INJECT_TAGS + "</body>", 1)
            else:
                html += _INJECT_TAGS
        content = html.encode("utf-8")
        _set_header(out_headers, "content-type", "text/html; charset=utf-8")
        _set_header(out_headers, "cache-control", "no-cache")

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=out_headers,
    )


def _register_root(app: FastAPI, has_webui: bool) -> None:
    """注册根路径 + ``/info``。

    平台「引擎操作」页与「在新窗口打开」都打**基础 URL**（``http://127.0.0.1:<port>/``）：
        - 有 WebUI → 307 跳转到 ``/ui/``，用户直接看到操作界面
        - 无 WebUI → 返回引擎信息 JSON（便于脚本/排障发现能力）
    ``/info`` 恒返回同一份 JSON（307 保留 method 语义不变）。
    """

    def _engine_info() -> dict:
        cfg = get_config()
        eng = get_engine()
        note = (
            "WebUI: GET /ui/（原生 UI 反代）；引擎信息: GET /info"
            if has_webui
            else "本引擎未提供 WebUI（webui/launcher.html 缺失），请通过平台任务中心或 API 调用"
        )
        return {
            "engine_id": cfg.engine_id,
            "engine_name": cfg.engine_name,
            "capability": cfg.capability,
            "type": cfg.engine_type,
            "version": "0.1.0",
            "backend": "strata.exe（外部二进制）",
            "mode": eng.process.current_mode or eng.loaded_mode,
            "running": eng.process.is_running,
            "variants": [v.name for v in eng._available_variants],
            "endpoints": [
                "GET /health", "GET /capabilities", "POST /load", "POST /unload",
                "POST /infer", "GET /status", "POST /lora/load", "POST /lora/unload",
                "GET /hardware", "POST /mode/switch", "GET /output/*",
                "GET|POST /log_level",
            ]
            + (["GET /ui/", "GET /info"] if has_webui else ["GET /"]),
            "webui": ("/ui/" if has_webui else None),
            "note": note,
        }

    @app.get("/info", include_in_schema=False)
    async def _info() -> dict:
        """引擎信息（恒可用）。"""
        return _engine_info()

    @app.get("/", include_in_schema=False)
    async def _root():
        """根路径：有 WebUI 则跳转，否则返回引擎信息。"""
        if has_webui:
            # 307 保留 method（脚本探测根路径时不被改写）
            return RedirectResponse(url="/ui/", status_code=307)
        return _engine_info()


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准端点的暴露层 + 2 个增强端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（6 个命名模式 + 推荐标记）。"""
        resp = await get_engine().capabilities()
        # 手动序列化 variants，确保 StrataVariantInfo 的扩展字段
        # （mode / mode_label / ctx_recommended / mtp / vision / ram_required_gb /
        #  low_vram_note / disk_required_gb / recommended / expected_cold_start_s）被保留。
        # CapabilitiesResponse.variants 声明类型为 list[VariantInfo]，
        # Pydantic model_dump() 按声明类型序列化会丢弃子类扩展字段。
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        return result

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """校验 bins + GGUF 分片（**只校验不下载**）→ 冷启动 strata.exe。

        请求体：``{"variant": "strata/balanced"}``
        资源缺失时返回 500 + 含"缺失文件 / 本地路径 / HF 仓库 / 操作方式"的错误。
        """
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """停止子进程 + 释放状态。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """对话生成（SSE 流式返回增量 token + 最终文本 URL）。

        请求体参考 InferRequest：
            - variant: 模式名（如 strata/balanced）
            - inputs : ChatInputs 字段（messages / prompt / max_tokens / …）
            - task_id: 可选任务 ID
        """
        engine = get_engine()
        sse_stream = engine.infer(request)
        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询（模式 / 内部端口 / pid / 硬件画像 / 可用模式）。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v1 预留，不支持）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（本引擎 v1 不支持，预留接口）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v1 预留，不支持）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（本引擎 v1 不支持，预留接口）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 增强端点（本引擎特有）=====

    # 9. /hardware：硬件探针 + 推荐模式
    @app.get("/hardware")
    async def hardware() -> dict:
        """硬件探针（GPU / 内存 / 磁盘 / CPU 特性）+ 推荐模式。

        探针任何子项失败都降级为「无法检测」，不阻断；结果缓存复用。
        """
        eng = get_engine()
        hw = eng.get_hardware()
        recommended = eng._recommend()
        return {
            "hardware": hw,
            "recommended_mode": recommended,
            # 低显存提示（⚠️ 实验性/慢；无提示时为 None），供 UI 展示
            "recommended_note": getattr(eng, "_recommend_note", None),
            "device": eng.config.resolve_device(),
            "spec": eng.config.spec,
            "modes": [
                {
                    "name": m["name"],
                    "label": m.get("label", ""),
                    "quant": m.get("quant_key", ""),
                    "ctx": m.get("ctx", 0),
                    "mtp": bool(m.get("mtp", False)),
                    "vision": bool(m.get("vision", False)),
                    "ram_required_gb": m.get("ram_required_gb", 0),
                    "disk_required_gb": m.get("disk_required_gb", 0),
                }
                for m in MODES
            ],
        }

    # 10. /mode/switch：冷切换模式（停 → 改写 → 冷启动，失败回滚）
    @app.post("/mode/switch")
    async def mode_switch(request: Request) -> dict:
        """冷切换模式。

        请求体：``{"mode": "strata/quality"}``（必填）
        切换时需重启引擎子进程（冷切换），失败自动回滚到旧模式。
        """
        try:
            body: dict[str, Any] = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="请求体必须为 JSON：{\"mode\": \"strata/xxx\"}")
        new_mode = (body or {}).get("mode")
        if not new_mode:
            raise HTTPException(status_code=400, detail="缺少必填字段 mode")
        try:
            result = await get_engine().switch_mode(str(new_mode))
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("模式切换失败: %s", new_mode)
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 通用 =====

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)

    # 根路径与 /info 由 ``_register_root`` 注册（需先知道 WebUI 是否挂载，
    # 因此不能放在这里 —— 见 create_app 中的调用顺序）。


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 Strata LLM 引擎: host=%s, port=%d, bins_dir=%s, model_dir=%s",
        _config.host, _config.port, _config.bins_dir, _config.model_dir,
    )

    app = create_app(_config)

    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()

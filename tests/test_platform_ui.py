"""``/ui/`` = 原生 Strata UI 别名前缀 —— 反代层单元测试。

覆盖（对应 server.py 的 ``_mount_platform_ui`` / ``_proxy_to_native``）：
    1. ``/ui`` → 307 ``/ui/``（尾斜杠，相对路径解析依赖它）
    2. 引擎未加载：``/ui/`` → 本地 launcher.html；其余路径 → 503
    3. 引擎已加载：``/ui/`` → 反代原生 index.html **且注入平台层两个标签**
    4. 相对路径资源透传：``/ui/web/app.js`` → 原生 ``/web/app.js``
    5. 原生 JSON 端点透传：``/ui/health``
    6. SSE（text/event-stream）**逐块透传不缓冲**
    7. ``/ui/_platform.js`` ``/ui/_platform.css`` 恒由本地提供（不反代）
    8. 根 ``/`` → 307 ``/ui/``（平台「引擎操作」页入口契约不变）
"""

from __future__ import annotations

import http.server
import json
import threading
import types

import pytest
from fastapi.testclient import TestClient

from engines.strata_llm_7c4f2b91 import server as srv
from engines.strata_llm_7c4f2b91.config import EngineConfig

# 原生 index.html 的最小等价物（含 </body>，用于验证注入位置）
_NATIVE_INDEX = (
    "<!doctype html><html><head><title>Strata</title>"
    '<link rel="stylesheet" href="web/tokens.css"></head><body>'
    "<header class=\"st-header\">Strata</header>"
    "<main></main><script src=\"web/app.js\"></script></body></html>"
)

_SSE_CHUNKS = [b'data: {"delta":"he"}\n\n', b'data: {"delta":"llo"}\n\n', b'data: [DONE]\n\n']


class _UpstreamHandler(http.server.BaseHTTPRequestHandler):
    """模拟 Strata 原生 serve/server.py 的最小路由集合。"""

    protocol_version = "HTTP/1.0"

    def log_message(self, *args):  # noqa: D102 - 静音测试输出
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server 约定
        path = self.path.split("?")[0]
        if path in ("", "/"):
            self._send(200, _NATIVE_INDEX.encode(), "text/html; charset=utf-8")
        elif path == "/web/app.js":
            self._send(200, b'console.log("native-app");', "text/javascript; charset=utf-8")
        elif path == "/health":
            self._send(200, json.dumps({"status": "ok", "service": "strata"}).encode(),
                       "application/json")
        elif path == "/metrics":
            self._send(200, json.dumps({"requests": []}).encode(), "application/json")
        else:
            self._send(404, b'{"error":{"message":"not found"}}', "application/json")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.split("?")[0] == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _SSE_CHUNKS:
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        self._send(404, b'{"error":{"message":"not found"}}', "application/json")


@pytest.fixture
def upstream():
    """启动一个本地原生桩服务，返回其 base url。"""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def cfg(tmp_path):
    return EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        output_root=tmp_path / "output",
        bins_dir=tmp_path / "bins",
        strata_data_dir=tmp_path / "data",
        strata_src_dir=tmp_path / "strata-src",
    )


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    """构造 app + TestClient（不进入 lifespan，避免真实引擎初始化）。"""
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    # 隔离模块级全局，保证每个用例从「未加载」开始
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(srv, "_config", cfg)
    app = srv.create_app(cfg)
    return TestClient(app)


def _fake_engine(url: str, running: bool = True):
    """最小可用引擎桩（``/ui`` 反代层与 ``/info`` 都会读它）。"""
    return types.SimpleNamespace(
        process=types.SimpleNamespace(
            is_running=running,
            internal_url=url if running else None,
            current_mode="strata/balanced" if running else None,
        ),
        loaded_mode="strata/balanced",
        _available_variants=[],
    )


# ===== 1. /ui → /ui/ =====

def test_ui_without_slash_redirects(client):
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


# ===== 2. 未加载兜底 =====

def test_launcher_when_not_loaded(client):
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "启动引擎" in body          # 启动器页特征
    assert "_platform.js" not in body  # 兜底页不依赖注入层
    # 必须 no-cache：否则引擎加载完成后浏览器会命中旧缓存继续显示启动器页
    assert r.headers.get("cache-control") == "no-cache"


def test_non_root_paths_503_when_not_loaded(client):
    assert client.get("/ui/web/app.js").status_code == 503
    assert client.get("/ui/health").status_code == 503


# ===== 3~4. 已加载：反代 + 注入 =====

def test_index_proxied_and_injected(client, upstream, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    r = client.get("/ui/")
    assert r.status_code == 200
    body = r.text
    # 原生内容透传
    assert "st-header" in body and "web/app.js" in body
    # 平台层注入
    assert srv._PLATFORM_CSS in body and srv._PLATFORM_JS in body
    assert body.count(srv._PLATFORM_JS) == 1        # 幂等：只注入一次
    assert body.index("web/app.js") < body.index(srv._PLATFORM_JS)  # 注入在原生脚本之后
    assert body.rstrip().endswith("</html>")
    assert r.headers.get("cache-control") == "no-cache"
    # 注入改写后不得留下重复 Content-Type（httpx 头名小写化，直接赋值会产生重复头）
    assert r.headers.get("content-type") == "text/html; charset=utf-8"


def test_relative_asset_proxied(client, upstream, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    r = client.get("/ui/web/app.js")
    assert r.status_code == 200
    assert "native-app" in r.text
    assert "javascript" in r.headers["content-type"]


def test_native_json_endpoint_proxied(client, upstream, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    r = client.get("/ui/health")
    assert r.status_code == 200
    assert r.json()["service"] == "strata"


def test_upstream_404_passthrough(client, upstream, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    assert client.get("/ui/nope").status_code == 404


def test_upstream_unreachable_502(client, monkeypatch):
    # 指向一个必然不可达的端口
    monkeypatch.setattr(srv, "_engine", _fake_engine("http://127.0.0.1:1"))
    r = client.get("/ui/health")
    assert r.status_code == 502


# ===== 6. SSE 不缓冲 =====

def test_sse_stream_proxied(client, upstream, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    with client.stream("POST", "/ui/v1/chat/completions", json={"model": "x"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_bytes())
    assert raw == b"".join(_SSE_CHUNKS)


# ===== 7. 注入层本地提供 =====

def test_platform_assets_are_local_even_when_unloaded(client, upstream, monkeypatch):
    r1 = client.get("/ui/_platform.js")
    r2 = client.get("/ui/_platform.css")
    assert r1.status_code == 200 and "javascript" in r1.headers["content-type"]
    assert r2.status_code == 200 and "css" in r2.headers["content-type"]
    assert "pf-bar" in r2.text            # 注入层样式特征
    assert r1.headers.get("cache-control") == "no-cache"
    assert r2.headers.get("cache-control") == "no-cache"
    # 即使引擎已加载，也必须命中本地文件而非上游
    monkeypatch.setattr(srv, "_engine", _fake_engine(upstream))
    assert client.get("/ui/_platform.js").status_code == 200


# ===== 8. 根路径契约 =====

def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


def test_info_always_available(client, monkeypatch):
    monkeypatch.setattr(srv, "_engine", _fake_engine("http://127.0.0.1:1", running=False))
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["webui"] == "/ui/"

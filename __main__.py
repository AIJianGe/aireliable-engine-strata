"""Strata LLM 引擎模块入口。

支持 ``python -m engines.strata_llm_7c4f2b91`` 直接启动。

启动命令（与全部既有引擎一致，走 -m 而非直接跑 server.py）：
    uv run --no-sync --project envs/llm-strata \\
        python -m engines.strata_llm_7c4f2b91.server --port 7010
"""

from .server import main

if __name__ == "__main__":
    main()

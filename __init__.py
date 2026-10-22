"""Strata LLM 推理引擎包。

基于 Strata（Niko1221/Strata，MIT）二进制运行时，为 AIReliable 平台提供
``chat`` 能力，实现平台 8 端点契约 + 6 个命名模式 + 硬件自适应推荐。

引擎 ID : strata_llm_7c4f2b91
端口    : 7010（注册片段 config/engines/strata_llm_7c4f2b91.yaml）
模型    : Qwen3.8-Flash-Next（125B MoE，GGUF 分片），落 models/llm/checkpoints/strata/

说明：
    - Python 层只做 8 端点契约 + 模式编排（冷切换）+ 原生 WebUI
    - 真正的推理由外部二进制 ``strata.exe``（STRATA_BINS_DIR 指定）承担
    - 不依赖 torch / diffusers（与 llama_cpp_9cdb76f8 的薄包装层同构）
"""

__version__ = "0.1.0"

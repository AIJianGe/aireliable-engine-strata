"""pytest 公共配置：确保项目根在 sys.path 中，使 ``engines.*`` 可导入。"""

from __future__ import annotations

import sys
from pathlib import Path

# tests → strata_llm_7c4f2b91 → engines → 项目根
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

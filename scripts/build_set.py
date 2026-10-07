"""造题入口。

    .venv\\Scripts\\python.exe scripts\\build_set.py --n 600

（等价于 python -m meval.build_set，但无需手动设置 PYTHONPATH。）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meval.build_set import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

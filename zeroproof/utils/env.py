"""
Environment and revision info collection helpers.

Used to record reproducibility context in result JSONs.
"""

import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _numpy_version() -> Optional[str]:
    try:
        import numpy as np  # type: ignore

        return str(np.__version__)
    except Exception:
        return None


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                info["device_count"] = torch.cuda.device_count()
                info["devices"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
            except Exception:
                pass
    except Exception:
        pass
    return info


def collect_env_info() -> Dict[str, Any]:
    """Collect environment info (Python, OS, NumPy/Torch, git)."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": _numpy_version(),
        "torch": _torch_info(),
        "git_commit": _git_commit_hash(),
    }

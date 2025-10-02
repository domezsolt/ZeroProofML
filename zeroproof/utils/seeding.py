"""Global seeding utilities for reproducibility.

Sets seeds for Python's `random`, NumPy, and PyTorch (if available),
and configures deterministic behavior where possible.
"""

import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Set seeds across libraries in a best-effort manner.

    Args:
        seed: Seed value. If None, does nothing.
    """
    if seed is None:
        return

    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    try:
        random.seed(seed)
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn  # type: ignore

            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

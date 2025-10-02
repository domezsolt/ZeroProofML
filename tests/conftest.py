"""Global test configuration and lightweight fixtures.

This file provides a minimal 'benchmark' fixture fallback so property
tests marked with @pytest.mark.benchmark run even when pytest-benchmark
is not installed. It also seeds RNGs for more deterministic behavior.
"""

import os
import random
from typing import Any, Callable
from pathlib import Path

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    """Seed common RNGs to improve test determinism."""
    seed = int(os.environ.get("ZPM_TEST_SEED", "12345"))
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
    except Exception:
        pass


class _DummyBenchmark:
    """Minimal stand-in for pytest-benchmark's fixture.

    Provides a callable interface and a 'pedantic' method that simply
    executes the function without timing. Suitable for tests that only
    need the API surface.
    """

    def __call__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def pedantic(self, func: Callable[[], Any], rounds: int = 1, *args: Any, **kwargs: Any) -> Any:
        result = None
        for _ in range(max(1, int(rounds))):
            result = func()
        return result


@pytest.fixture
def benchmark() -> _DummyBenchmark:
    """Provide a basic benchmark fixture when pytest-benchmark is absent."""
    return _DummyBenchmark()


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests under tests/property with the 'property' marker.

    CI selects property tests via `-m property`. Some tests in tests/property
    don't explicitly carry the marker, so ensure they are selectable.
    """
    for item in items:
        try:
            p = Path(str(item.fspath))
            # Mark any test file within a 'tests/property' directory tree
            if any(part == "property" for part in p.parts) and "tests" in p.parts:
                item.add_marker(pytest.mark.property)  # type: ignore[attr-defined]
        except Exception:
            # Best-effort; ignore path issues
            pass

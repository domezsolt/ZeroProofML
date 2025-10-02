from zeroproof.utils.seeding import set_global_seed


def test_seeding_reproducible():
    # Seed once and sample
    set_global_seed(123)
    try:
        import numpy as np
    except Exception:
        # If numpy not available, pass basic check using random
        import random

        set_global_seed(123)
        a = [random.random() for _ in range(5)]
        set_global_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b
        return

    a1 = np.random.rand(5).tolist()
    set_global_seed(123)
    a2 = np.random.rand(5).tolist()
    assert a1 == a2

"""Tests for NumPy TRArrayPacked bit-packed representation."""

import numpy as np

from zeroproof.bridge.numpy_bridge import TRArray, TRArrayPacked, count_tags
from zeroproof.bridge.numpy_bridge import from_numpy as from_numpy_std
from zeroproof.bridge.numpy_bridge import from_numpy_packed, to_numpy


def test_packed_roundtrip_matches_standard():
    arr = np.array([0.0, 1.0, -2.5, np.inf, -np.inf, np.nan, 3.14], dtype=np.float64)
    std = from_numpy_std(arr)
    packed = from_numpy_packed(arr)

    assert isinstance(std, TRArray)
    assert isinstance(packed, TRArrayPacked)

    # to_numpy should reconstruct the same IEEE array semantics
    out_std = to_numpy(std)
    out_packed = to_numpy(packed)

    # Compare elementwise with NaN-aware equality
    for a, b in zip(out_std, out_packed):
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert a == b

    # Tags count should match
    counts_std = count_tags(std)
    # For packed, reconstruct TRArray for counting via standard API
    # (count_tags expects TRArray; emulate by converting back/forth)
    # Alternatively, reconstruct masks and compare
    real_mask = packed.is_real()
    pinf_mask = packed.is_pinf()
    ninf_mask = packed.is_ninf()
    phi_mask = packed.is_phi()
    counts_packed = {
        "REAL": int(real_mask.sum()),
        "PINF": int(pinf_mask.sum()),
        "NINF": int(ninf_mask.sum()),
        "PHI": int(phi_mask.sum()),
    }
    assert counts_std == counts_packed

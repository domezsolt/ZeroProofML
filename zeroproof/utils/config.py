"""
Central configuration values to avoid drift across components.
"""

# Default |det(J)| bucket edges (B0 includes exact zeros)
# B0: [0, 1e-5], B1: (1e-5, 1e-4], B2: (1e-4, 1e-3], B3: (1e-3, 1e-2], B4: (1e-2, inf)
DEFAULT_BUCKET_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]

# Canonical loss identifier for comparators
LOSS_NAME = "mse_mean"

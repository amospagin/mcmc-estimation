"""Online adaptation mechanisms.

All adaptation is online (no separate warmup phase) and cross-chain:
  - Step size: dual averaging
  - Mass matrix: Welford online covariance
  - Flow parameters: score matching (see transforms.score_matching)
"""

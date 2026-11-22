"""Final custom-loss LSTMs (MAE + JSD + smoothness, after arXiv:2510.16898).

All three share inputs, the one-layer LSTM with price skip branch (EPEX lags skip-only), the loss
and the ERA5 pretraining (``common.py``); they differ only in the daily update:

* ``pretrained_only``: no updates.
* ``rolling_window``: fine-tuning on the last 7 complete days.
* ``adaptive``: training on the new day, accepted only if the RMSE on the
  6 previous days improves.

The intermediate iterations that led here live in ``pipeline/mlp``.
"""

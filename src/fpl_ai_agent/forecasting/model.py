"""Temporal CNN model for next-gameweek FPL point forecasting."""

from __future__ import annotations


def _require_torch():
    """Import torch lazily with a clear error message if unavailable."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for forecasting. Install torch in the active env.") from exc
    return torch, nn


class TemporalCNNRegressor:
    """Factory wrapper that constructs the real torch.nn.Module lazily."""

    @staticmethod
    def build(
        *,
        input_features: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Build a temporal CNN module returning predictive mean and variance."""
        torch, nn = _require_torch()

        class _TemporalCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                padding = kernel_size // 2
                self.encoder = nn.Sequential(
                    nn.Conv1d(input_features, conv_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head_mean = nn.Linear(conv_channels, 1)
                self.head_logvar = nn.Linear(conv_channels, 1)

            def forward(self, x):
                # Input shape is (batch, window_length, features), transpose for Conv1d.
                z = x.transpose(1, 2)
                z = self.encoder(z).squeeze(-1)
                mean = self.head_mean(z).squeeze(-1)
                logvar = self.head_logvar(z).squeeze(-1)
                variance = torch.nn.functional.softplus(logvar) + 1e-6
                return mean, variance

        return _TemporalCNN()

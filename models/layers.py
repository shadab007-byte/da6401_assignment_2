"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """custom Dropout layer implementing.

      - Inverted dropout scales activations by 1/(1-p) during training so that
        the expected value of each activation remains the same at test time,
        requiring no rescaling during inference.
      - use torch.bernoulli on a uniform random tensor rather than any
        built-in dropout utility, as required by the assignment.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor with same shape as input.
        """
        if not self.training or self.p == 0.0:
            return x

        # Create binary mask: keep probability = 1 - p
        keep_prob = 1.0 - self.p
        # torch.bernoulli samples 1 with probability given by the input tensor
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype))
        # Inverted dropout: scale by 1/keep_prob so test-time expectation is unchanged
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"

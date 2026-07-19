import math

import torch
import torch.nn as nn


def _find_nearest_divisor(value: int, target: int) -> int:
    """Find the largest divisor of value that is <= target."""
    for d in range(target, 0, -1):
        if value % d == 0:
            return d
    return 1



def _find_divisor_pair(target: int):
    """Find a balanced pair of divisors (d, target//d) for the target."""
    best = (1, target)
    for d in range(int(math.sqrt(target)), 0, -1):
        if target % d == 0:
            return d, target // d
    return best


class LoRAErrorCompensation(nn.Module):
    """Low-rank error compensation for quantized weights.

    This module approximates the quantization error
    E = W - Q(W)
    using a Kronecker product of two learnable matrices:

        E \approx A \otimes B

    where A and B are much smaller than the full weight matrix.
    """

    def __init__(
        self,
        weight_shape,
        a_shape=None,
        b_shape=None,
        rank=None,
        device=None,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.weight_shape = tuple(weight_shape)
        self.a_shape, self.b_shape = self._infer_kron_shapes(
            self.weight_shape, a_shape, b_shape, rank
        )
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        self.A = nn.Parameter(torch.zeros(self.a_shape, device=self.device))
        self.B = nn.Parameter(torch.randn(self.b_shape, device=self.device))
        nn.init.normal_(self.B, mean=0.0, std=init_scale)


    def _infer_kron_shapes(self, weight_shape, a_shape, b_shape, rank):
        H, W = weight_shape
        if a_shape is not None and b_shape is not None:
            if a_shape[0] * b_shape[0] != H or a_shape[1] * b_shape[1] != W:
                raise ValueError(
                    f"Invalid kron shapes: {a_shape} x {b_shape} -> {a_shape[0] * b_shape[0]} x {a_shape[1] * b_shape[1]},"
                    f" expected {H} x {W}"
                )
            return tuple(a_shape), tuple(b_shape)

        if a_shape is not None:
            if H % a_shape[0] != 0 or W % a_shape[1] != 0:
                raise ValueError(
                    f"a_shape={a_shape} is not compatible with weight shape {weight_shape}"
                )
            b_shape = (H // a_shape[0], W // a_shape[1])
            return tuple(a_shape), tuple(b_shape)

        if b_shape is not None:
            if H % b_shape[0] != 0 or W % b_shape[1] != 0:
                raise ValueError(
                    f"b_shape={b_shape} is not compatible with weight shape {weight_shape}"
                )
            a_shape = (H // b_shape[0], W // b_shape[1])
            return tuple(a_shape), tuple(b_shape)

        if rank is not None:
            if H % rank == 0 and W % rank == 0:
                return (H // rank, W // rank), (rank, rank)

            if H % rank == 0:
                return (H // rank, W), (rank, 1)

            if W % rank == 0:
                return (H, W // rank), (1, rank)

            r_h, r_w = _find_divisor_pair(rank)
            if H % r_h == 0 and W % r_w == 0:
                return (H // r_h, W // r_w), (r_h, r_w)
            if H % r_w == 0 and W % r_h == 0:
                return (H // r_w, W // r_h), (r_w, r_h)

            raise ValueError(
                f"rank={rank} cannot be converted to valid Kronecker factors for weight shape {weight_shape}"
            )

        a_rows = _find_nearest_divisor(H, int(math.sqrt(H)))
        a_cols = _find_nearest_divisor(W, int(math.sqrt(W)))
        b_rows = H // a_rows
        b_cols = W // a_cols
        return (a_rows, a_cols), (b_rows, b_cols)

    def reset_parameters(self, init_scale: float = 1e-3):
        nn.init.normal_(self.A, mean=0.0, std=init_scale)
        nn.init.normal_(self.B, mean=0.0, std=init_scale)

    def compensation(self) -> torch.Tensor:
        """Return the full compensation matrix A \otimes B."""
        return torch.kron(self.A, self.B)

    def forward(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Apply compensation to quantized weight."""
        error = self.compensation()
        if error.shape != quantized_weight.shape:
            raise ValueError(
                f"Compensation shape {error.shape} does not match quantized weight shape {quantized_weight.shape}"
            )
        #print(f"max compensation: {error.abs().max().item()}")
        return quantized_weight + error

    def save_state_dict(self) -> dict:
        return {
            "A": self.A.detach().cpu(),
            "B": self.B.detach().cpu(),
            "a_shape": self.a_shape,
            "b_shape": self.b_shape,
            "device": str(self.device),
        }

    
__all__ = ["LoRAErrorCompensation"]

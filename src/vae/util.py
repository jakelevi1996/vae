import torch

LOG_2_PI = torch.log(torch.tensor(2 * torch.pi))

ZERO = torch.tensor(0.0)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, ZERO)

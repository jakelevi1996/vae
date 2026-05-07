import torch

class ShiftedCircle:
    def __init__(
        self,
        inner_rad:      float=2.0,
        outer_rad:      float=3.0,
        left_shift:     float=-1.0,
        right_shift:    float=1.0,
    ):
        self.inner_rad      = inner_rad
        self.thickness      = outer_rad - inner_rad
        self.left_shift     = left_shift
        self.right_shift    = right_shift

    def sample(self, n: int) -> torch.Tensor:
        theta_n = 2 * torch.pi * torch.rand([n])
        r_n = self.inner_rad + self.thickness * torch.rand([n])

        x1_n = r_n * torch.cos(theta_n)
        x2_n = r_n * torch.sin(theta_n)

        x_n2 = torch.stack([x1_n, x2_n], -1)

        x_n2[x_n2[:, 0] < 0] += torch.tensor([0, self.left_shift])
        x_n2[x_n2[:, 0] > 0] += torch.tensor([0, self.right_shift])

        return x_n2

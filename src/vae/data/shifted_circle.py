import torch

class ShiftedCircle:
    def __init__(
        self,
        n_train:        int=1000,
        n_test:         int=1000,
        inner_rad:      float=2.0,
        outer_rad:      float=3.0,
        left_shift:     float=-1.0,
        right_shift:    float=1.0,
    ):
        self.x_train_ni = self.get_data(
            n=n_train,
            inner_rad=inner_rad,
            outer_rad=outer_rad,
            left_shift=left_shift,
            right_shift=right_shift,
        )
        self.x_test_ni = self.get_data(
            n=n_test,
            inner_rad=inner_rad,
            outer_rad=outer_rad,
            left_shift=left_shift,
            right_shift=right_shift,
        )

    def get_data(
        self,
        n:              int,
        inner_rad:      float,
        outer_rad:      float,
        left_shift:     float,
        right_shift:    float,
    ):
        theta_n = 2 * torch.pi * torch.rand([n])
        r_n = inner_rad + (outer_rad - inner_rad) * torch.rand([n])

        x1_n = r_n * torch.cos(theta_n)
        x2_n = r_n * torch.sin(theta_n)

        x_n2 = torch.stack([x1_n, x2_n], -1)

        x_n2[x_n2[:, 0] < 0] += torch.tensor([0, left_shift])
        x_n2[x_n2[:, 0] > 0] += torch.tensor([0, right_shift])

        return x_n2

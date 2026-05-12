import torch
import juml
from vae.models.multi_head_mlp import MultiHeadReluMlp
from vae.util import softplus, LOG_2_PI

class Vae(juml.base.Model):
    def __init__(
        self,
        input_dim:          int,
        latent_dim:         int,
        hidden_dim:         int,
        num_hidden_layers:  int,
    ):
        self._torch_module_init()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.log_sigma_z = torch.nn.Parameter(torch.zeros([latent_dim]))
        self.log_sigma_x = torch.nn.Parameter(torch.zeros([input_dim]))

        self.encoder = MultiHeadReluMlp(
            input_dim=input_dim,
            output_dims=[latent_dim],
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.decoder = MultiHeadReluMlp(
            input_dim=latent_dim,
            output_dims=[input_dim],
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.opt = torch.optim.AdamW(self.parameters())

    def step(self, x_ni: torch.Tensor) -> tuple[float, float, float]:
        [mu_z_nh] = self.encoder.forward(x_ni)
        sigma_z_nh = self.log_sigma_z.exp()

        kl_components = (
            1
            + 2*sigma_z_nh.log()
            - mu_z_nh.square()
            - sigma_z_nh.square()
        )
        kl_term = 0.5 * kl_components.sum(-1).mean()

        eps_nh = torch.normal(0, 1, [*x_ni.shape[:-1], self.latent_dim])
        z_nh = mu_z_nh + eps_nh * sigma_z_nh

        [mu_x_ni] = self.decoder.forward(z_nh)
        sigma_x_ni = self.log_sigma_x.exp()

        reconstruct_components = (
            - LOG_2_PI
            - 2*sigma_x_ni.log()
            - ((x_ni - mu_x_ni) / sigma_x_ni).square()
        )
        reconstruct_term = 0.5 * reconstruct_components.sum(-1).mean()

        elbo = kl_term + reconstruct_term

        self.opt.zero_grad()
        (-elbo).backward()
        self.opt.step()

        return elbo.item(), kl_term.item(), reconstruct_term.item()

    def sample(self, n: int) -> torch.Tensor:
        z_nh = torch.normal(0, 1, [n, self.latent_dim])

        [mu_x_ni] = self.decoder.forward(z_nh)
        sigma_x_ni = self.log_sigma_x.exp()

        eps_ni = torch.normal(0, 1, [n, self.input_dim])

        x_ni = mu_x_ni + eps_ni * sigma_x_ni

        return x_ni

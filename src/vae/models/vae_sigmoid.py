import torch
import juml
from vae.models.vae import Vae
from vae.util import softplus, LOG_2_PI

class VaeSigmoid(Vae):
    def step(self, x_ni: torch.Tensor) -> tuple[float, float, float]:
        mu_z_nh, sigma_logit_z_nh = self.encoder.forward(x_ni)
        sigma_z_nh = softplus(sigma_logit_z_nh) + self.eps

        kl_components = (
            1
            + 2*sigma_z_nh.log()
            - mu_z_nh.square()
            - sigma_z_nh.square()
        )
        kl_term = 0.5 * kl_components.sum(-1).mean()

        eps_nh = torch.normal(0, 1, [*x_ni.shape[:-1], self.latent_dim])
        z_nh = mu_z_nh + eps_nh * sigma_z_nh

        mu_x_ni, sigma_logit_x_ni = self.decoder.forward(z_nh)
        mu_x_ni = mu_x_ni.sigmoid()
        sigma_x_ni = softplus(sigma_logit_x_ni) + self.eps

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

        mu_x_ni, sigma_logit_x_ni = self.decoder.forward(z_nh)
        mu_x_ni = mu_x_ni.sigmoid()
        sigma_x_ni = softplus(sigma_logit_x_ni) + self.eps

        eps_ni = torch.normal(0, 1, [n, self.input_dim])

        x_ni = mu_x_ni + eps_ni * sigma_x_ni

        return x_ni

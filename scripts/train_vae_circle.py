import torch
from jutility import plotting, util, cli
import juml
import vae

def main():
    seed = 0

    latent_dim = 2
    hidden_dim = 100
    num_hidden_layers = 1

    steps = 100000
    batch_size = 100
    plot_samples = 1000

    torch.manual_seed(seed)

    dataset = vae.data.ShiftedCircle()

    model = vae.models.Vae(
        input_dim=2,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )

    table = util.Table(
        util.CountColumn(),
        util.TimeColumn(),
        util.Column("elbo",             ".5f"),
        util.Column("kl_term",          ".5f"),
        util.Column("reconstruct_term", ".5f"),
        print_interval=util.TimeInterval(1),
    )

    for _ in range(steps):
        x_ni = dataset.sample(batch_size)
        elbo, kl_term, reconstruct_term = model.step(x_ni)
        table.update(
            elbo=elbo,
            kl_term=kl_term,
            reconstruct_term=reconstruct_term,
        )

    x_data = dataset.sample(plot_samples)
    x_model = model.sample(plot_samples).detach()

    cp = plotting.ColourPicker.contrast()

    output_dir = "results/train_vae_circle"

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *[
                plotting.Line(table.get_data(s), c=c, label=s)
                for c, s in zip(cp, "elbo kl_term reconstruct_term".split())
            ],
            plotting.Legend(),
        ),
        plotting.Subplot(
            plotting.Scatter(
                x_data[:, 0],
                x_data[:, 1],
                color=cp(0),
                label="Data",
                z=10,
                a=0.4,
            ),
            plotting.Scatter(
                x_model[:, 0],
                x_model[:, 1],
                color=cp(1),
                label="Model",
                z=20,
                a=0.4,
            ),
            plotting.Legend(),
        ),
        fs=[10, 4],
    )
    mp.save(dir_name=output_dir)

if __name__ == "__main__":
    with util.Timer("main"):
        main()

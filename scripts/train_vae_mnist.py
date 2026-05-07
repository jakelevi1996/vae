import torch
from jutility import plotting, util, cli
import juml
import vae

def main(
    args:   cli.ParsedArgs,
    seed:   int,
    epochs: int,
):
    latent_dim = 10
    hidden_dim = 200
    num_hidden_layers = 1

    batch_size = 100
    plot_samples = 25

    torch.manual_seed(seed)

    dataset = vae.data.Mnist()
    data_loader = dataset.get_data_loader("train", batch_size)

    model = vae.models.VaeSigmoid(
        input_dim=dataset.get_input_dim(),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )

    table = util.Table(
        util.CountColumn(),
        util.TimeColumn(),
        util.Column("epoch",            "i"),
        util.Column("batch",            "i"),
        util.Column("elbo",             ".5f"),
        util.Column("kl_term",          ".5f"),
        util.Column("reconstruct_term", ".5f"),
        print_interval=util.TimeInterval(1),
    )

    for e in range(epochs):
        for b, (x, _) in enumerate(data_loader):
            x = dataset.format_x(x)
            elbo, kl_term, reconstruct_term = model.step(x)
            table.update(
                epoch=e,
                batch=b,
                elbo=elbo,
                kl_term=kl_term,
                reconstruct_term=reconstruct_term,
            )

    x_model = model.sample(plot_samples).detach().clip(0, 1)

    cp = plotting.ColourPicker.contrast()

    output_dir = "results/train_vae_mnist/%s" % args.get_summary()

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *[
                plotting.Line(table.get_data(s), c=c, label=s)
                for c, s in zip(cp, "elbo kl_term reconstruct_term".split())
            ],
            plotting.Legend(),
        ),
        plotting.MultiPlot(
            *[
                plotting.Subplot(
                    plotting.ImShow(x_model[i].reshape(28, 28)),
                )
                for i in range(plot_samples)
            ],
        ),
        fs=[10, 4],
    )
    mp.save(dir_name=output_dir)

if __name__ == "__main__":
    parser = cli.Parser(
        cli.Arg("seed",     type=int, default=0),
        cli.Arg("epochs",   type=int, default=100),
    )
    args = parser.parse_args()

    with util.Timer("main"):
        main(args, **args.get_kwargs())

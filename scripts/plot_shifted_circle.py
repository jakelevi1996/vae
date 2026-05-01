from jutility import plotting, util, cli
import vae

def main():
    dataset = vae.data.ShiftedCircle()

    x_n2 = dataset.x_train_ni

    plotting.plot(
        plotting.Scatter(x_n2[:, 0], x_n2[:, 1], c="b", a=0.4),
        axis_equal=True,
        plot_name="Shifted Circle dataset",
    )

if __name__ == "__main__":
    with util.Timer("main"):
        main()

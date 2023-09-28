import matplotlib.pyplot as plt


def plot_convergence(curves):
    labels = {'opt_gap': r"$\epsilon_{\mathrm{opt}}$", 'resid': r"$\epsilon_{\mathrm{cert}}$"}
    colors = {'Vaidya': 'b', 'Ellipsoids': 'g'}
    line_styles = {('Vaidya', 'opt_gap'): '-', ('Vaidya', 'resid'): (0, (1, 1)),
                   ('Ellipsoids', 'opt_gap'): (0, (6, 2)), ('Ellipsoids', 'resid'): '-.'}

    plt.rcParams['axes.grid'] = True
    plt.rcParams.update({'font.size': 16})

    n_reg = len(curves)
    n_dim = len(next(iter(curves.items()))[1])
    fig, axs = plt.subplots(n_dim, n_reg, figsize=(8 * n_reg, 5 * n_dim))

    for j, reg in enumerate(curves.keys()):
        axs[0, j].set_title(fr"$\mu=${reg}")
        for i, n in enumerate(curves[reg].keys()):
            axs[i, j].set_yscale('log')
            if j == 0:
                axs[i, 0].get_yaxis().set_label_coords(-0.18, 0.5)
                axs[i, 0].set_ylabel(fr"$n=${n}", rotation=0, size='large')
            for algo in ['Vaidya', 'Ellipsoids']:
                for curve_type in ['opt_gap', 'resid']:
                    curve = curves[reg][n][algo][curve_type]
                    axs[i, j].plot(range(1, len(curve)+1), curve, linestyle=line_styles[(algo, curve_type)],
                                   color=colors[algo], linewidth=2, label=algo + ', ' + labels[curve_type])
                    axs[i, j].set_xlabel("Oracle calls")

    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = fig.legend(handles, labels, bbox_to_anchor=[0.5, 0.95], loc='upper center', ncol=4)

    fig.subplots_adjust(hspace=0.25)
    plt.savefig(f"Fig1.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

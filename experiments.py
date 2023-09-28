import numpy as np
import pickle

from ellipsoid import ellipsoid_method, get_certificate, get_residual_and_solution
from utils import plot_convergence
from vaidya import Vaidya


def objective(x, reg):
    return x.max() + reg * np.linalg.norm(x)**2 / 2


def subgradient(x, reg):
    coord_vec = np.zeros_like(x)
    coord_vec[x.argmax()] = 1
    return coord_vec + reg * x


def exact_solution(n, reg):
    solution = -np.ones(n) / (reg * n)
    Opt = -1. / (2 * reg * n)
    return solution, Opt


def get_residuals_and_solutions(protocol, productive, R):
    residuals = []
    solutions = []
    n_iter = len(protocol) - 1
    for i in range(1, n_iter + 1):
        xis = get_certificate(protocol[:i+1], productive[:i])
        residual, solution = get_residual_and_solution(xis, protocol[:i+1], R, productive[:i])
        residuals.append(residual)
        solutions.append(solution.copy())

    return residuals, solutions


def experiment():
    reg = 0.1

    # Vaidya parameters
    eps = 5e-3
    tau = 1.
    n_newton = 5

    curves = dict()
    for reg in [0.01, 0.1]:
        print(f"reg={reg}")
        curves[reg] = dict()
        for n in [10, 20, 30]:
            print(f"n={n}")
            curves[reg][n] = dict()
            n_iter = n * 50

            solution, Opt = exact_solution(n, reg)
            R = 10 * np.linalg.norm(solution)
            oracle = lambda x: (subgradient(x, reg), True)

            protocol_ellipse, productive_ellipse = ellipsoid_method(R, n, oracle, n_iter)
            resid_ellipse, solutions_ellipse = get_residuals_and_solutions(protocol_ellipse, productive_ellipse, R)
            eps_opt_ellipse = [(objective(x, reg) - Opt) for x in solutions_ellipse]
            curves[reg][n]['Ellipsoids'] = {'resid': resid_ellipse, 'opt_gap': eps_opt_ellipse}

            vaidya_alg = Vaidya(R, n, oracle, eps, tau, n_newton)
            residuals, solutions = vaidya_alg.iterate(n_iter)
            eps_opt = [(objective(x, reg) - Opt) for x in solutions]
            curves[reg][n]['Vaidya'] = {'resid': residuals, 'opt_gap': eps_opt}

    with open(f"curves.pickle", 'wb') as handle:
        pickle.dump(curves, handle)


def main():
    experiment()

    with open(f"curves.pickle", 'rb') as handle:
        curves = pickle.load(handle)

    plot_convergence(curves)


if __name__ == '__main__':
    main()

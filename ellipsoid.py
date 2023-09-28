import numpy as np
import matplotlib.pyplot as plt


def alpha_gamma(n):
    return n / np.sqrt(n**2 - 1), n / (n + 1)


def ellipsoid_method(R, n, oracle, n_iter):
    """
    :param R: radius of a unit ball centered at 0 containing the feasible set
    :param n: dimension of the problem, n > 1
    :param oracle: function implementing 1st order and separation oracles
    :param n_iter: number of iterations
    :param plot:
    :return:
    """
    B = R * np.eye(n)
    c = np.zeros(n)
    alp, gam = alpha_gamma(n)

    protocol = []
    productive = np.zeros(n_iter, dtype=bool)
    for i in range(n_iter):
        e, prod = oracle(c)
        protocol.append((B.copy(), c.copy(), e.copy()))
        productive[i] = prod

        if np.all(e == 0):
            print("zero grad")
            break

        BTq = B.T @ e
        p = BTq / np.linalg.norm(BTq)
        c -= B @ p / (n + 1)
        B = alp * B + (gam - alp) * B @ p[:, np.newaxis] @ p[np.newaxis, :]

    protocol.append((B.copy(), c.copy()))
    plt.show()
    return protocol, productive


def get_certificate(protocol, productive):
    if np.all(productive == 0):
        print("No productive steps")
        return None

    e_tau = protocol[-2][2]
    if np.all(e_tau == 0):
        print("Terminal step")
        xis = np.zeros(len(protocol) - 1)
        xis[-1] = 1
        return xis

    lambdas, mus = [], []
    B, c = protocol[-1][0], protocol[-1][1]
    h = get_narrowest_stripe(B)
    h_plus = (h, -h @ c)
    h_minus = (-h, h @ c)

    for B, c, e in protocol[-2::-1]:
        h_plus, lam = decompose(h_plus, B, c, e)
        lambdas.append(lam)
        h_minus, mu = decompose(h_minus, B, c, e)
        mus.append(mu)

    lambdas = np.array(lambdas[::-1])
    mus = np.array(mus[::-1])
    d = (lambdas + mus) @ productive
    if d == 0:
        print("Certificate not defined: d = 0")
        return None

    xis = (lambdas + mus) / d
    return xis


def get_residual_and_solution(xis, protocol, R, productive):
    if xis is None:
        return None, None

    es = np.column_stack([Bce[2] for Bce in protocol[:-1]])
    resid_maximizer = es @ xis
    resid_maximizer *= -R / np.linalg.norm(resid_maximizer)

    cs = np.column_stack([Bce[1] for Bce in protocol[:-1]])
    residual = (xis * es * (cs - resid_maximizer[:, np.newaxis])).sum()
    solution = cs @ (xis * productive)
    return residual, solution


def get_narrowest_stripe(B):
    U, D, _ = np.linalg.svd(B)
    i_star = D.argmin()
    return U[:, i_star] / (2 * D[i_star])


def decompose(h_plus, B, c, e):
    g, a = h_plus
    p, q = B.T @ g, B.T @ e
    r = p @ q / (q @ q)
    r = r if r > 0 else 0

    f = g - r * e
    b = a + r * c @ e
    return (f, b), r

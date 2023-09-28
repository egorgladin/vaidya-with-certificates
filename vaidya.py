import numpy as np
from scipy.optimize import linprog


def inv(mat):
    try:
        mat_inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        mat_inv = np.linalg.pinv(mat)
    return mat_inv


class Vaidya:
    def __init__(self, R, n, oracle, eps, tau, n_newton):
        self.x = np.zeros(n)
        self.oracle = oracle
        self.tau = tau
        self.eps = eps
        self.n_newton = n_newton
        self.R = R

        self.trajectory = []
        self.grads = []
        self.protocol = []
        self.iteration = 0
        self.init_polytope(n, R)
        self.do_update()

    def init_polytope(self, n, R):
        self.A = np.vstack((np.eye(n), -np.eye(n)))
        self.b = R * np.ones(2 * n)
        self.t_of_i = np.zeros(2 * n, dtype=int)

    def upd_slacks(self):
        self.s = self.b - self.A @ self.x

    def upd_SinvA(self):
        self.SinvA = (self.s**(-1))[:, None] * self.A

    def upd_Hinv(self):
        self.Hinv = inv(self.SinvA.T @ self.SinvA)

    def upd_P(self):
        self.P = self.SinvA @ self.Hinv @ self.SinvA.T

    def upd_sigmas(self):
        self.sigmas = np.diag(self.P)

    def add_row(self, a_new, b_new):
        self.A = np.vstack((self.A, a_new.T))
        self.b = np.append(self.b, b_new)
        self.t_of_i = np.append(self.t_of_i, self.iteration)

    def remove_row(self, i):
        self.A = np.delete(self.A, i, 0)
        self.b = np.delete(self.b, i)
        self.t_of_i = np.delete(self.t_of_i, i)

    def do_update(self):
        self.upd_slacks()
        self.upd_SinvA()
        self.upd_Hinv()
        self.upd_P()
        self.upd_sigmas()

    def upd_x(self):
        for step in range(self.n_newton):
            hess = self.SinvA.T @ (3 * np.diag(self.sigmas) - 2 * self.P**2) @ self.SinvA
            self.x -= inv(hess) @ (self.SinvA.T @ self.sigmas)
            self.do_update()

    def step(self):
        self.iteration += 1

        a_new, _ = self.oracle(self.x)  # e_t = F'(x_t)
        self.trajectory.append(self.x.copy())
        self.grads.append(a_new.copy())
        b_new = np.dot(a_new, self.x) + np.sqrt((a_new.T @ self.Hinv @ a_new) / self.tau)

        self.add_row(a_new, b_new)  # Q_t -> \hat{Q}_{t+1}
        self.do_update()
        self.upd_x()

        while (self.sigmas < self.eps).any():
            i = self.sigmas.argmin()
            self.remove_row(i)  # Q_{t+1} contains \hat{Q}_{t+1}
            self.do_update()
            self.upd_x()  # x_{t+1} \in int Q_{t+1}

    def iterate(self, n_iter):
        residuals, solutions = [], []

        for i in range(n_iter):
            self.step()
            self.certify()
            residuals.append(self.resid)
            solutions.append(self.x_tau)

        return residuals, solutions

    def certify(self):
        prod = (self.t_of_i != 0)
        c = -np.linalg.norm(self.A * prod[:, None], axis=1)
        A_eq = self.A.T
        b_eq = np.zeros(len(self.x))
        A_ub = np.row_stack((self.b, -self.b))
        b_ub = np.array([2., 0.])
        lam = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs').x

        lam_sum = lam[prod].sum()
        if lam_sum <= 0:
            print("Certificate is not defined")
            self.xi = None
            self.x_tau = None
            self.resid = None

        else:
            self.xi = lam[prod] / lam_sum

            nnz_t_of_i = self.t_of_i[prod] - 1
            ordered_traj = np.column_stack(self.trajectory)[:, nnz_t_of_i]  # (n, m)
            self.x_tau = ordered_traj @ self.xi

            ordered_grads = np.column_stack(self.grads)[:, nnz_t_of_i]
            e_tau = ordered_grads @ self.xi
            resid_maximizer = -self.R * np.sign(e_tau)  # (n,)
            self.resid = (ordered_grads * (ordered_traj - resid_maximizer[:, None]) * self.xi).sum()

# cutting_plane_generic.py
import numpy as np
from scipy.optimize import linprog


class CuttingPlane:
    """
    Cutting-plane method (Kelley 1959) para f convexa (possivelmente não suave).

    Parâmetros
    ----------
    f        : callable(x) -> float
               devolve f(x)
    subgrad  : callable(x) -> ndarray (n,)
               devolve um subgradiente em x  (elemento de ∂f(x))
    bounds   : list[ tuple(min,max) ]
               caixa de busca D = \prod_i [l_i,u_i]
               Use (None,None) para variável não-limitada.
    x0       : ponto inicial (iterável de tamanho n)
    tol      : para |f(x) − \check f_k(x)| <= tol  ⇒ convergiu
    max_iter : máximo de iterações
    """

    def __init__(self, f, subgrad, bounds, x0, tol=1e-4, max_iter=200):
        self.f, self.subgrad = f, subgrad
        self.bounds = bounds
        self.x0 = np.asarray(x0, dtype=float)
        self.tol = tol
        self.max_iter = max_iter

        self.n = len(self.x0)  # dimensão
        self.bundle = []  # lista de tuplas (x_i, f_i, s_i)
        self.history = []  # (x, f, t_lower, gap)

    # ------------------------------------------------------------------
    # passo interno: resolve   min_{x∈D}  max_{(f_i,s_i)}  g_i(x)
    # usando LP: variável extra t representa o máximo
    # ------------------------------------------------------------------
    def _solve_subproblem(self):
        m = len(self.bundle)  # #cortes
        if m == 0:
            raise RuntimeError("Bundle vazio")

        # Variáveis LP = (x_1,...,x_n, t)  ⇒ total n+1
        c = np.zeros(self.n + 1)
        c[-1] = 1.0  # min  t

        # Desigualdades:   f_i + s_i^T (x - x_i)  <=  t
        A_ub, b_ub = [], []
        for x_i, f_i, s_i in self.bundle:
            a = np.zeros(self.n + 1)
            a[: self.n] = s_i  #  s_i^T x
            a[-1] = -1.0  # -t
            A_ub.append(a)
            b_ub.append(-f_i + s_i @ x_i)  # - f_i + s_i^T x_i

        # Bounds: herdamos caixa D e adicionamos bound para t (None,None)
        bounds_ext = list(self.bounds) + [(None, None)]

        res = linprog(
            c,
            A_ub=np.asarray(A_ub),
            b_ub=np.asarray(b_ub),
            bounds=bounds_ext,
            method="highs",
        )

        if not res.success:
            raise RuntimeError("LP interno não convergiu: " + res.message)

        x_new = res.x[: self.n]
        t_new = res.x[-1]
        return x_new, t_new

    # ------------------------------------------------------------------
    # loop principal
    # ------------------------------------------------------------------
    def solve(self):
        xk = self.x0.copy()
        for k in range(self.max_iter):
            fk = self.f(xk)
            sk = self.subgrad(xk)

            # guarda corte
            self.bundle.append((xk, fk, sk))

            # resolve sub-LP
            x_new, t_new = self._solve_subproblem()

            gap = abs(self.f(x_new) - t_new)
            self.history.append((x_new, self.f(x_new), t_new, gap))

            if gap < self.tol:
                print(f"Parou na iteração {k+1}  |  gap = {gap:.2e}")
                return x_new

            xk = x_new

        raise RuntimeError("Máx. de iterações atingido sem convergir")


# ----------------------------------------------------------------------
# EXEMPLO 1-D  —  f(x) = |x|  em [-5,5]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    f = lambda x: np.abs(x)[0]  # linprog precisa array-like
    sg = lambda x: (
        np.array([1.0])
        if x[0] > 0
        else (np.array([-1.0]) if x[0] < 0 else np.array([0.0]))
    )

    solver = CuttingPlane(f, sg, bounds=[(-5, 5)], x0=[4.0])
    x_star = solver.solve()
    print(f"x* = {x_star},  f(x*) = {f(x_star):.4f}")

    # histórico (opcional)
    for i, (x, fval, t, gap) in enumerate(solver.history, 1):
        print(
            f"it {i:2d}:  x = {x},  f = {fval:.4f},  lower = {t:.4f},  gap = {gap:.2e}"
        )

import numpy as np
from scipy.linalg import lu_factor, lu_solve  # opcional, veja abaixo


def solve_system(A, b, method="numpy"):
    """
    Resolve Ax = b.

    Parâmetros
    ----------
    A : array‑like ou np.ndarray (m, m)
        Matriz dos coeficientes.
    b : array‑like ou np.ndarray (m,)
        Vetor do lado direito.
    method : str, {"numpy", "lu"}, default "numpy"
        • "numpy" → usa np.linalg.solve (mais simples).
        • "lu"    → fatoração LU (útil para muitos b diferentes).

    Retorna
    -------
    x : np.ndarray (m,)
        Solução do sistema.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if method == "numpy":
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as err:
            raise ValueError(f"Sistema não resolvido: {err}") from None

    elif method == "lu":
        try:
            lu, piv = lu_factor(A)  # decompõe A = LU
            x = lu_solve((lu, piv), b)  # resolve LUx=b
        except Exception as err:
            raise ValueError(f"Problema na fatoração LU: {err}") from None
    else:
        raise ValueError("Método desconhecido. Use 'numpy' ou 'lu'.")

    return x


# ----------------- Exemplo rápido -----------------
if __name__ == "__main__":
    # Sistema:  2x +  y −  z =  8
    #           −3x − y + 2z = −11
    #           −2x + y + 2z = −3
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]

    sol_np = solve_system(A, b, method="numpy")
    sol_lu = solve_system(A, b, method="lu")

    print("Solução (np.linalg.solve):", sol_np)
    # print("Solução (LU):            ", sol_lu)

    print("Questao 1:")
    # Sistema:  2x  - 0y − l =  4
    #           0x + 2y + lcosl = 0
    #           -x + seny + 0l = 0

    A = [[2, 0, -1], [0, 2, np.cos(1)], [-1, np.sin(1), 0]]
    b = [4, 0, 0]

    sol_np = solve_system(A, b, method="numpy")
    print("Solução (np.linalg.solve):", sol_np)

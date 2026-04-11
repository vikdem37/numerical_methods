import numpy as np


def generate_matrix(n):
    return np.random.uniform(-100, 100, (n, n))


def save_matrix(path, M):
    np.savetxt(path, M, fmt="%.15e")


def load_matrix(path):
    return np.loadtxt(path)


def save_vector(path, v):
    np.savetxt(path, v, fmt="%.15e")


def load_vector(path):
    return np.loadtxt(path)


def lu_decompose(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)

    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]

    return L, U


def solve_lu(L, U, b):
    n = len(b)
    z = np.zeros(n)
    x = np.zeros(n)

    z[0] = b[0] / L[0, 0]
    for k in range(1, n):
        z[k] = (b[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]

    x[n - 1] = z[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))

    return x


def mat_vec(A, x):
    return A @ x


def norm_inf(v):
    return np.max(np.abs(v))


if __name__ == "__main__":
    np.random.seed(42)
    n = 100

    A = generate_matrix(n)
    x_true = np.full(n, 2.5)
    B = mat_vec(A, x_true)

    save_matrix("lab6/matrix_A.txt", A)
    save_vector("lab6/vector_B.txt", B)
    print(f"Згенеровано матрицю A ({n}x{n}) та вектор B ({n})")

    A = load_matrix("lab6/matrix_A.txt")
    B = load_vector("lab6/vector_B.txt")
    print("Зчитано A та B з файлів")

    L, U = lu_decompose(A)
    save_matrix("lab6/matrix_L.txt", L)
    save_matrix("lab6/matrix_U.txt", U)

    lu_err = norm_inf(L @ U - A)
    print(f"\nLU-розклад: ||L*U - A|| = {lu_err:.4e}")

    X = solve_lu(L, U, B)

    eps = norm_inf(mat_vec(A, X) - B)
    err_x = norm_inf(X - x_true)
    print(f"\nТочність: eps = max|A*X - B| = {eps:.4e}")
    print(f"Похибка розв'язку: max|X - x_true| = {err_x:.4e}")

    eps0 = 1e-14
    max_iter = 100
    print(f"\nІтераційне уточнення (eps0 = {eps0:.0e}):")
    print(f"  {'Ітер':>4}  {'||dX||':>14}  {'||AX-B||':>14}")
    print("  " + "-" * 36)

    for it in range(1, max_iter + 1):
        R = B - mat_vec(A, X)
        dX = solve_lu(L, U, R)
        X = X + dX

        norm_dx = norm_inf(dX)
        norm_res = norm_inf(mat_vec(A, X) - B)
        print(f"  {it:4d}  {norm_dx:14.4e}  {norm_res:14.4e}")

        if norm_dx <= eps0 and norm_res <= eps0:
            print(f"\nЗбіжність за {it} ітерацій")
            break
    else:
        print(f"\nНе збіглося за {max_iter} ітерацій")

    eps_final = norm_inf(mat_vec(A, X) - B)
    err_final = norm_inf(X - x_true)
    print(f"Фінальна точність: eps = {eps_final:.4e}")
    print(f"Фінальна похибка: max|X - x_true| = {err_final:.4e}")

    print("\nФайли: lab6/matrix_A.txt, lab6/vector_B.txt, lab6/matrix_L.txt, lab6/matrix_U.txt")

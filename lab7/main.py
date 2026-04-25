import numpy as np


def generate_diag_dominant(n):
    A = np.random.uniform(-10, 10, (n, n))
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 10)
    return A


def save_matrix(path, M):
    np.savetxt(path, M, fmt="%.15e")


def load_matrix(path):
    return np.loadtxt(path)


def save_vector(path, v):
    np.savetxt(path, v, fmt="%.15e")


def load_vector(path):
    return np.loadtxt(path)


def mat_vec(A, x):
    return A @ x


def norm_inf(v):
    return np.max(np.abs(v))


def matrix_norm_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))


def simple_iteration(A, f, x0, tau, eps, max_iter):
    x = x0.copy()
    n = len(x)
    E = np.eye(n)
    C = E - tau * A
    d = tau * f

    for k in range(1, max_iter + 1):
        x_new = C @ x + d
        dx = norm_inf(x_new - x)
        res = norm_inf(A @ x_new - f)
        x = x_new
        if dx <= eps and res <= eps:
            return x, k

    return x, max_iter


def jacobi(A, f, x0, eps, max_iter):
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diag(D)

    for k in range(1, max_iter + 1):
        x_new = (f - R @ x) / D
        dx = norm_inf(x_new - x)
        res = norm_inf(A @ x_new - f)
        x = x_new
        if dx <= eps and res <= eps:
            return x, k

    return x, max_iter


def gauss_seidel(A, f, x0, eps, max_iter):
    x = x0.copy()
    n = len(x)

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (f[i] - s1 - s2) / A[i, i]

        dx = norm_inf(x - x_old)
        res = norm_inf(A @ x - f)
        if dx <= eps and res <= eps:
            return x, k

    return x, max_iter


if __name__ == "__main__":
    np.random.seed(42)
    n = 100

    A = generate_diag_dominant(n)
    x_true = np.full(n, 2.5)
    B = mat_vec(A, x_true)

    save_matrix("lab7/matrix_A.txt", A)
    save_vector("lab7/vector_B.txt", B)
    print(f"Згенеровано матрицю A ({n}x{n}) з діагональним переважанням та вектор B ({n})")

    A = load_matrix("lab7/matrix_A.txt")
    B = load_vector("lab7/vector_B.txt")
    print("Зчитано A та B з файлів")

    A_norm = matrix_norm_inf(A)
    print(f"\n||A||_inf = {A_norm:.4f}")

    tau = 1.0 / A_norm
    print(f"tau = 1/||A||_inf = {tau:.6e}  (в межах (0, 2/||A||) = (0, {2 / A_norm:.6e}))")

    x0 = np.ones(n)
    eps0 = 1e-14
    max_iter = 10000

    print("\nПочаткове наближення: x_i^(0) = 1.0")
    print(f"Задана точність: eps0 = {eps0:.0e}")
    print(f"Максимальне число ітерацій: {max_iter}")

    x_si, it_si = simple_iteration(A, B, x0, tau, eps0, max_iter)
    res_si = norm_inf(mat_vec(A, x_si) - B)
    err_si = norm_inf(x_si - x_true)
    print("\nМетод простої ітерації:")
    print(f"  Ітерацій: {it_si}")
    print(f"  ||AX - B|| = {res_si:.4e}")
    print(f"  max|X - x_true| = {err_si:.4e}")
    print(f"  X[:5]  = {x_si[:5]}")
    print(f"  X[-5:] = {x_si[-5:]}")

    x_j, it_j = jacobi(A, B, x0, eps0, max_iter)
    res_j = norm_inf(mat_vec(A, x_j) - B)
    err_j = norm_inf(x_j - x_true)
    print("\nМетод Якобі:")
    print(f"  Ітерацій: {it_j}")
    print(f"  ||AX - B|| = {res_j:.4e}")
    print(f"  max|X - x_true| = {err_j:.4e}")
    print(f"  X[:5]  = {x_j[:5]}")
    print(f"  X[-5:] = {x_j[-5:]}")

    x_gs, it_gs = gauss_seidel(A, B, x0, eps0, max_iter)
    res_gs = norm_inf(mat_vec(A, x_gs) - B)
    err_gs = norm_inf(x_gs - x_true)
    print("\nМетод Гауса-Зейделя:")
    print(f"  Ітерацій: {it_gs}")
    print(f"  ||AX - B|| = {res_gs:.4e}")
    print(f"  max|X - x_true| = {err_gs:.4e}")
    print(f"  X[:5]  = {x_gs[:5]}")
    print(f"  X[-5:] = {x_gs[-5:]}")

    print(f"\n{'Метод':<25} {'Ітерацій':>10} {'||AX-B||':>14} {'Похибка':>14}")
    print("-" * 65)
    print(f"{'Простої ітерації':<25} {it_si:>10d} {res_si:>14.4e} {err_si:>14.4e}")
    print(f"{'Якобі':<25} {it_j:>10d} {res_j:>14.4e} {err_j:>14.4e}")
    print(f"{'Гауса-Зейделя':<25} {it_gs:>10d} {res_gs:>14.4e} {err_gs:>14.4e}")

    print("\nФайли: lab7/matrix_A.txt, lab7/vector_B.txt")

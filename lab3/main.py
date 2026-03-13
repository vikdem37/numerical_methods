import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    x, y = [], []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))
    return np.array(x), np.array(y)


def form_matrix(x, m):
    n = m + 1
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * x ** i)
    return b


def gauss_solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol


def polynomial(x, coef):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for i, c in enumerate(coef):
        result += c * x ** i
    return result


def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)


if __name__ == "__main__":
    x, y = read_data("lab3/data.csv")
    print("Дані: Місяць, Температура")
    for i in range(len(x)):
        print(f"  {x[i]:5.0f}  {y[i]:6.1f}")

    max_degree = 4 # pdf says 10, pseudocode says 4
    # i mean tbf i'd rather live under -68.36C than 510.33C
    variances = []
    all_coefs = []
    print()
    print(f"  {'m':>3}  {'Дисперсія':>14}")
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)
        all_coefs.append(coef)
        print(f"  {m:3d}  {var:14.4f}")

    optimal_m = np.argmin(variances) + 1
    print(f"\nОптимальний степінь: m = {optimal_m}")

    coef = all_coefs[optimal_m - 1]
    print("Коефіцієнти:")
    for i, c in enumerate(coef):
        print(f"  a_{i} = {c:.6f}")

    y_approx = polynomial(x, coef)

    n = len(x)
    x0, xn = x[0], x[-1]
    h1 = (xn - x0) / (20 * n)
    x_fine = np.arange(x0, xn + h1 / 2, h1)
    y_interp = np.interp(x_fine, x, y)

    error_fine = {}
    for m in range(1, max_degree + 1):
        phi = polynomial(x_fine, all_coefs[m - 1])
        error_fine[m] = np.abs(y_interp - phi)

    print(f"\nТабуляція похибки (h1 = {h1:.4f}):")
    header = f"{'x':>8}"
    for m in range(1, max_degree + 1):
        header += f" | {'m=' + str(m):>10}"
    print(header)
    print("-" * len(header))
    for j in range(len(x_fine)):
        row = f"{x_fine[j]:>8.3f}"
        for m in range(1, max_degree + 1):
            row += f" | {error_fine[m][j]:>10.4f}"
        print(row)

    with open("lab3/tabulation.txt", "w", encoding="utf-8") as f:
        f.write(f"Похибка апроксимації |f(x) - φ(x)| для m = 1..{max_degree}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for j in range(len(x_fine)):
            row = f"{x_fine[j]:>8.3f}"
            for m in range(1, max_degree + 1):
                row += f" | {error_fine[m][j]:>10.4f}"
            f.write(row + "\n")

    x_future = np.array([25, 26, 27])
    y_future = polynomial(x_future, coef)
    print("\nПрогноз:")
    for i in range(len(x_future)):
        print(f"  Місяць {x_future[i]:.0f}: {y_future[i]:.2f} °C")

    xx = np.linspace(x0, x_future[-1], 300)
    yy = polynomial(xx, coef)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "bo", markersize=6, label="Експериментальні дані")
    plt.plot(xx, yy, "g-", lw=2, label=f"Апроксимація (m={optimal_m})")
    plt.plot(x_future, y_future, "rs", markersize=8, label="Прогноз")
    plt.xlabel("Місяць")
    plt.ylabel("Температура (°C)")
    plt.title("Апроксимація температури методом найменших квадратів")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab3/fig_approximation.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    for m in range(1, max_degree + 1):
        plt.plot(x_fine, error_fine[m], lw=1.5, label=f"m = {m}")
    plt.xlabel("x")
    plt.ylabel("|f(x) - φ(x)|")
    plt.title("Похибка апроксимації для різних степенів")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab3/fig_errors.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    degrees = list(range(1, max_degree + 1))
    colors = ["steelblue" if d != optimal_m else "orangered" for d in degrees]
    plt.figure(figsize=(7, 4))
    plt.bar(degrees, variances, color=colors)
    plt.xlabel("Степінь полінома m")
    plt.ylabel("Дисперсія")
    plt.title("Дисперсія апроксимації залежно від степеня")
    plt.xticks(degrees)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("lab3/fig_variance.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print()
    print("Графіки: lab3/fig_*.png")
    print("Табуляція: lab3/tabulation.txt")

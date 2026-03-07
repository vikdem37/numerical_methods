import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    x, y = [], []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return np.array(x), np.array(y)


def build_divided_differences(x, y):
    n = len(x)
    dd = np.zeros((n, n))
    dd[:, 0] = y
    for k in range(1, n):
        for i in range(n - k):
            dd[i, k] = (dd[i + 1, k - 1] - dd[i, k - 1]) / (x[i + k] - x[i])
    return dd


def newton_value(x, y, x_val):
    dd = build_divided_differences(x, y)
    n = len(x)
    p = dd[0, 0]
    prod = 1.0
    for k in range(1, n):
        prod *= x_val - x[k - 1]
        p += dd[0, k] * prod
    return p


def factorial_value(x, y, x_val):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return y[0] if len(y) else None
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        return None
    n = len(x)
    d = np.zeros((n, n))
    d[:, 0] = y
    for k in range(1, n):
        for i in range(n - k):
            d[i, k] = d[i + 1, k - 1] - d[i, k - 1]
    q = (x_val - x[0]) / h
    p = d[0, 0]
    term = 1.0
    for k in range(1, n):
        term *= (q - (k - 1)) / k
        p += d[0, k] * term
    return p


def lagrange_value(x, y, x_val):
    n = len(x)
    result = 0.0
    for i in range(n):
        basis = 1.0
        for j in range(n):
            if j != i:
                basis *= (x_val - x[j]) / (x[i] - x[j])
        result += y[i] * basis
    return result


def newton_on_grid(x_nodes, y_nodes, xx):
    return np.array([newton_value(x_nodes, y_nodes, xi) for xi in xx])


def omega_on_grid(x_nodes, xx):
    result = np.ones_like(xx, dtype=float)
    for xi in x_nodes:
        result *= (xx - xi)
    return result


def runge_function(x, center=8500, half_width=7500):
    t = (x - center) / half_width
    return 1.0 / (1.0 + 25.0 * t ** 2)


def print_dd_table(x, y, dd):
    n = len(x)
    print("Таблиця розділених різниць")
    print("i |    x_i    | f[x_i]  ", end="")
    for k in range(1, n):
        print(f" |   dd_{k}   ", end="")
    print()
    for i in range(n):
        row = f"{i} | {x[i]:9.1f} | {dd[i, 0]:7.2f}"
        for k in range(1, n):
            cell = f"{dd[i, k]:10.4g}" if i <= n - 1 - k else ""
            row += f" | {cell:>10}"
        print(row)


if __name__ == "__main__":
    x, y = read_data("lab2/data.csv")
    print("Дані: n, t")
    for i in range(len(x)):
        print(f"  {x[i]:.0f}  {y[i]:.0f}")

    with open("lab2/tabulation.txt", "w", encoding="utf-8") as f:
        f.write("Табуляція вузлів\n")
        f.write("  i |     n     |   t (мс)\n")
        for i in range(len(x)):
            f.write(f"  {i} | {x[i]:9.1f} | {y[i]:8.2f}\n")

    num_tab = 10
    x_tab_nodes = np.linspace(x[0], x[-1], num_tab)
    y_tab_nodes = newton_on_grid(x, y, x_tab_nodes)
    xx_tab = np.linspace(x[0], x[-1], 30)
    yy_tab_ref = newton_on_grid(x, y, xx_tab)
    yy_tab_pn = newton_on_grid(x_tab_nodes, y_tab_nodes, xx_tab)
    eps_tab = yy_tab_ref - yy_tab_pn
    omega_tab = omega_on_grid(x_tab_nodes, xx_tab)
    with open("lab2/tabulation.txt", "a", encoding="utf-8") as f:
        f.write(f"\nТабуляція на відрізку (P з {num_tab} рівновіддалених вузлів)\n")
        f.write(f"  {'x':>10}  {'f(x)':>10}  {'P_n(x)':>10}  {'eps(x)':>12}  {'omega(x)':>14}\n")
        for i in range(len(xx_tab)):
            f.write(f"  {xx_tab[i]:10.1f}  {yy_tab_ref[i]:10.4f}  {yy_tab_pn[i]:10.4f}  {eps_tab[i]:12.4e}  {omega_tab[i]:14.4e}\n")

    dd = build_divided_differences(x, y)
    print()
    print_dd_table(x, y, dd)

    x_pred = 6000
    p_newton = newton_value(x, y, x_pred)
    p_lagrange = lagrange_value(x, y, x_pred)
    p_fact = factorial_value(x, y, x_pred)
    print()
    print(f"P(6000) (Ньютон)   = {p_newton:.4f}")
    print(f"P(6000) (Лагранж)  = {p_lagrange:.4f}")
    if p_fact is not None:
        print(f"P(6000) (факторіальний) = {p_fact:.4f}")
    else:
        print("Факторіальний многочлен: вузли не рівновіддалені")
    print(f"Різниця Ньютон-Лагранж = {abs(p_newton - p_lagrange):.2e}")

    a, b = x[0], x[-1]
    xx = np.linspace(a, b, 300)
    yy_ref = newton_on_grid(x, y, xx)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "bo", markersize=8, label="Експериментальні точки")
    plt.plot(xx, yy_ref, "b-", lw=2, label="Інтерполяційний многочлен Ньютона")
    plt.xlabel("n")
    plt.ylabel("t (мс)")
    plt.title("Прогноз часу виконання алгоритму")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab2/fig_newton_interp.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    num_sub = 10
    x_sub = np.linspace(a, b, num_sub)
    y_sub = newton_on_grid(x, y, x_sub)
    yy_sub = newton_on_grid(x_sub, y_sub, xx)
    err_sub = yy_ref - yy_sub
    omega_sub = omega_on_grid(x_sub, xx)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(xx, yy_ref, "b-", lw=2, label="Задана функція (5 вузлів)")
    axes[0].plot(x, y, "bo", markersize=6)
    axes[0].set_ylabel("t (мс)")
    axes[0].set_title("f(x) — задана функція")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xx, yy_sub, "g-", lw=2, label=f"Інтерполянт ({num_sub} вузлів)")
    axes[1].plot(x_sub, y_sub, "go", markersize=5)
    axes[1].set_ylabel("t (мс)")
    axes[1].set_title(f"P_n(x) — наближення ({num_sub} рівновіддалених вузлів)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xx, err_sub, "r-", lw=1, label="ε(x) = f(x) − P_n(x)")
    axes[2].axhline(0, color="k", ls="--", alpha=0.5)
    axes[2].set_ylabel("Похибка")
    axes[2].set_title("ε(x) — похибка інтерполяції")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(xx, omega_sub, "m-", lw=1, label="ω(x) = ∏(x − xᵢ)")
    axes[3].axhline(0, color="k", ls="--", alpha=0.5)
    axes[3].set_xlabel("n")
    axes[3].set_ylabel("ω(x)")
    axes[3].set_title("ω(x) — вузловий многочлен")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab2/fig_error_analysis.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print()
    print("Вплив кількості вузлів (рівновіддалені, [1000, 16000])")
    print(f"  {'n':>3}  {'max|err|':>10}  {'mean|err|':>10}  {'P(6000) Newton':>15}  {'P(6000) факт':>15}")
    fig_ni, ax_ni = plt.subplots(figsize=(10, 5))
    for num_nodes in [5, 10, 20]:
        x_eq = np.linspace(a, b, num_nodes)
        y_eq = newton_on_grid(x, y, x_eq)
        yy_eq = newton_on_grid(x_eq, y_eq, xx)
        err = yy_ref - yy_eq
        p_n = newton_value(x_eq, y_eq, x_pred)
        p_f = factorial_value(x_eq, y_eq, x_pred)
        p_f_str = f"{p_f:.4f}" if p_f is not None else "N/A"
        print(f"  {num_nodes:3d}  {np.max(np.abs(err)):10.4f}  {np.mean(np.abs(err)):10.4f}  {p_n:15.4f}  {p_f_str:>15}")
        ax_ni.plot(xx, err, lw=1.5, label=f"n={num_nodes}")
    ax_ni.axhline(0, color="k", ls="--", alpha=0.5)
    ax_ni.set_xlabel("n")
    ax_ni.set_ylabel("Похибка")
    ax_ni.set_title("Похибка інтерполяції для різної к-ті вузлів")
    ax_ni.legend()
    ax_ni.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_ni.savefig("lab2/fig_node_influence.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print()
    h_fixed = (b - a) / 20
    print(f"Фіксований крок h={h_fixed:.0f}, змінний інтервал [a, a+n*h]")
    print(f"  {'n':>3}  {'інтервал':>20}  {'вузлів':>6}  {'max|err|':>10}  {'mean|err|':>10}")
    for n_int in [5, 10, 20]:
        b_end = a + n_int * h_fixed
        n_nodes = n_int + 1
        x_fi = np.linspace(a, b_end, n_nodes)
        y_fi = newton_on_grid(x, y, x_fi)
        xx_fi = np.linspace(a, b_end, 200)
        yy_ref_fi = newton_on_grid(x, y, xx_fi)
        yy_fi = newton_on_grid(x_fi, y_fi, xx_fi)
        err_fi = yy_ref_fi - yy_fi
        print(f"  {n_int:3d}  [{a:.0f}, {b_end:.0f}]{'':>8}  {n_nodes:6d}  {np.max(np.abs(err_fi)):10.4f}  {np.mean(np.abs(err_fi)):10.4f}")

    print()
    print("Ефект Рунге (тестова ф-ція f(x)=1/(1+25t^2), t=(x-8500)/7500)")
    xx_runge = np.linspace(a, b, 500)
    yy_runge_true = runge_function(xx_runge)

    fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 5))
    axes_r[0].plot(xx_runge, yy_runge_true, "k-", lw=2, label="f(x) точна")
    print(f"  {'n':>3}  {'max|err|':>10}")
    for num_nodes in [5, 10, 20]:
        x_rn = np.linspace(a, b, num_nodes)
        y_rn = runge_function(x_rn)
        yy_rn = newton_on_grid(x_rn, y_rn, xx_runge)
        max_err = np.max(np.abs(yy_runge_true - yy_rn))
        print(f"  {num_nodes:3d}  {max_err:10.4f}")
        axes_r[0].plot(xx_runge, yy_rn, lw=1, label=f"P(x), n={num_nodes}")

    axes_r[0].set_xlabel("x")
    axes_r[0].set_ylabel("f(x)")
    axes_r[0].set_title("Ефект Рунге: інтерполяція")
    axes_r[0].legend(fontsize=8)
    axes_r[0].grid(True, alpha=0.3)

    for num_nodes in [5, 10, 20]:
        x_rn = np.linspace(a, b, num_nodes)
        y_rn = runge_function(x_rn)
        yy_rn = newton_on_grid(x_rn, y_rn, xx_runge)
        axes_r[1].plot(xx_runge, yy_runge_true - yy_rn, lw=1, label=f"n={num_nodes}")
    axes_r[1].axhline(0, color="k", ls="--", alpha=0.5)
    axes_r[1].set_xlabel("x")
    axes_r[1].set_ylabel("Похибка")
    axes_r[1].set_title("Ефект Рунге: похибка")
    axes_r[1].legend(fontsize=8)
    axes_r[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig_r.savefig("lab2/fig_runge.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print()
    print("Графіки: lab2/fig_*.png")
    print("Табуляція: lab2/tabulation.txt")

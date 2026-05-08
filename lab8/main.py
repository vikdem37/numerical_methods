import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return np.cos(x)


def F1(x):
    return -np.sin(x)


def F2(x):
    return -np.cos(x)


def tabulate(a, b, h, path):
    xs = np.arange(a, b + h / 2, h)
    ys = F(xs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'x':>10}  {'F(x)':>14}\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:10.4f}  {y:14.8f}\n")
    return xs, ys


def find_sign_changes(xs, ys):
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            kind = "зростаюча" if ys[i + 1] > ys[i] else "спадна"
            roots.append((xs[i], xs[i + 1], kind))
    return roots


def simple_iteration(x0, tau, eps, max_iter):
    x = x0
    for k in range(1, max_iter + 1):
        x_new = x + tau * F(x)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def newton(x0, eps, max_iter):
    x = x0
    for k in range(1, max_iter + 1):
        x_new = x - F(x) / F1(x)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def chebyshev(x0, eps, max_iter):
    x = x0
    for k in range(1, max_iter + 1):
        f, f1, f2 = F(x), F1(x), F2(x)
        x_new = x - f / f1 - 0.5 * f * f * f2 / f1**3
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def chord(x0, x1, eps, max_iter):
    for k in range(1, max_iter + 1):
        y0, y1 = F(x0), F(x1)
        x_new = x1 - y1 * (x1 - x0) / (y1 - y0)
        if abs(F(x_new)) < eps and abs(x_new - x1) < eps:
            return x_new, k
        x0, x1 = x1, x_new
    return x1, max_iter


def parabola(x0, x1, x2, eps, max_iter):
    for k in range(1, max_iter + 1):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        f01 = (y1 - y0) / (x1 - x0)
        f12 = (y2 - y1) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + f012 * (x2 - x1)
        disc = w * w - 4 * y2 * f012
        if disc < 0:
            disc = 0
        sq = np.sqrt(disc)
        denom = w + sq if abs(w + sq) > abs(w - sq) else w - sq
        delta = -2 * y2 / denom
        x_new = x2 + delta
        if abs(F(x_new)) < eps and abs(x_new - x2) < eps:
            return x_new, k
        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


def inverse_interp(x0, x1, x2, eps, max_iter):
    for k in range(1, max_iter + 1):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        x_new = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0 \
              + (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1 \
              + (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        if abs(F(x_new)) < eps and abs(x_new - x2) < eps:
            return x_new, k
        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


def save_coefficients(path, coefs):
    with open(path, "w", encoding="utf-8") as f:
        for c in coefs:
            f.write(f"{c:.15e}\n")


def read_coefficients(path):
    with open(path, "r", encoding="utf-8") as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def poly_eval(coefs, x):
    m = len(coefs) - 1
    result = coefs[m]
    for i in range(m - 1, -1, -1):
        result = result * x + coefs[i]
    return result


def horner_b(coefs, x):
    m = len(coefs) - 1
    b = np.zeros(m + 1)
    b[m] = coefs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coefs[i] + x * b[i + 1]
    return b


def horner_c(b_coefs, x):
    m = len(b_coefs) - 1
    c = np.zeros(m + 1)
    c[m] = b_coefs[m]
    for i in range(m - 1, 0, -1):
        c[i] = b_coefs[i] + x * c[i + 1]
    return c


def newton_horner(coefs, x0, eps, max_iter):
    x = x0
    for k in range(1, max_iter + 1):
        b = horner_b(coefs, x)
        c = horner_c(b, x)
        x_new = x - b[0] / c[1]
        if abs(b[0]) < eps and abs(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def lin_method(coefs, alpha0, beta0, eps, max_iter):
    m = len(coefs) - 1
    a = coefs.astype(float).copy()

    for k in range(1, max_iter + 1):
        p0 = -2 * alpha0
        q0 = alpha0**2 + beta0**2

        b = np.zeros(m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p0 * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p0 * b[i + 1] - q0 * b[i + 2]

        if abs(b[2]) < 1e-15:
            return alpha0, beta0, k

        q1 = a[0] / b[2]
        p1 = (a[1] * b[2] - a[0] * b[3]) / (b[2]**2)

        alpha1 = -p1 / 2
        rad = q1 - alpha1**2
        if rad < 0:
            rad = 0
        beta1 = np.sqrt(rad)

        if abs(alpha1 - alpha0) < eps and abs(beta1 - beta0) < eps:
            return alpha1, beta1, k
        alpha0, beta0 = alpha1, beta1

    return alpha0, beta0, max_iter


if __name__ == "__main__":
    a, b = -2.0, 2.0
    h = 0.1
    eps = 1e-10
    max_iter = 1000

    xs, ys = tabulate(a, b, h, "lab8/tabulation.txt")
    print(f"F(x) = cos(x), інтервал [{a}, {b}], крок h = {h}")
    print(f"Табуляція збережена у lab8/tabulation.txt ({len(xs)} вузлів)")

    brackets = find_sign_changes(xs, ys)
    print(f"\nЗнайдено {len(brackets)} перетин(ів) з віссю Ox:")
    for x_l, x_r, kind in brackets:
        x_mid = (x_l + x_r) / 2
        print(f"  [{x_l:.2f}, {x_r:.2f}]  ({kind}), наближення x = {x_mid:.4f}")

    root1_bracket = brackets[0]
    root2_bracket = brackets[1]
    x1_init = (root1_bracket[0] + root1_bracket[1]) / 2
    x2_init = (root2_bracket[0] + root2_bracket[1]) / 2
    print(f"\nПочаткові наближення: x1 = {x1_init:.4f}, x2 = {x2_init:.4f}")
    print(f"Точні значення коренів: -pi/2 = {-np.pi / 2:.10f}, pi/2 = {np.pi / 2:.10f}")

    results = []

    tau1 = -1.0 / F1(x1_init)
    tau2 = -1.0 / F1(x2_init)
    r1, n1 = simple_iteration(x1_init, tau1, eps, max_iter)
    r2, n2 = simple_iteration(x2_init, tau2, eps, max_iter)
    results.append(("Простої ітерації", r1, n1, r2, n2))

    r1, n1 = newton(x1_init, eps, max_iter)
    r2, n2 = newton(x2_init, eps, max_iter)
    results.append(("Ньютона", r1, n1, r2, n2))

    r1, n1 = chebyshev(x1_init, eps, max_iter)
    r2, n2 = chebyshev(x2_init, eps, max_iter)
    results.append(("Чебишева", r1, n1, r2, n2))

    r1, n1 = chord(root1_bracket[0], root1_bracket[1], eps, max_iter)
    r2, n2 = chord(root2_bracket[0], root2_bracket[1], eps, max_iter)
    results.append(("Хорд", r1, n1, r2, n2))

    r1, n1 = parabola(root1_bracket[0], (root1_bracket[0] + root1_bracket[1]) / 2,
                      root1_bracket[1], eps, max_iter)
    r2, n2 = parabola(root2_bracket[0], (root2_bracket[0] + root2_bracket[1]) / 2,
                      root2_bracket[1], eps, max_iter)
    results.append(("Парабол", r1, n1, r2, n2))

    r1, n1 = inverse_interp(root1_bracket[0], (root1_bracket[0] + root1_bracket[1]) / 2,
                            root1_bracket[1], eps, max_iter)
    r2, n2 = inverse_interp(root2_bracket[0], (root2_bracket[0] + root2_bracket[1]) / 2,
                            root2_bracket[1], eps, max_iter)
    results.append(("Зворотної інтерполяції", r1, n1, r2, n2))

    print(f"\nЗадана точність: eps = {eps:.0e}")
    print(f"\n{'Метод':<25} {'Корінь 1':>14} {'Ітер':>5}   {'Корінь 2':>14} {'Ітер':>5}")
    print("-" * 75)
    for name, r1, n1, r2, n2 in results:
        print(f"{name:<25} {r1:>14.10f} {n1:>5d}   {r2:>14.10f} {n2:>5d}")

    x_plot = np.linspace(a, b, 500)
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, F(x_plot), "b-", lw=2, label="F(x) = cos(x)")
    plt.axhline(y=0, color="k", lw=0.8)
    plt.plot([-np.pi / 2, np.pi / 2], [0, 0], "ro", markersize=10, label="Корені")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Трансцендентне рівняння F(x) = cos(x) = 0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab8/fig_transcendental.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\n")
    print("Алгебраїчне рівняння P(x) = x^3 - 4x^2 + 9x - 10 = 0")
    print("Корені: дійсний x = 2, комплексні x = 1 +/- 2i")

    coefs = np.array([-10.0, 9.0, -4.0, 1.0])
    save_coefficients("lab8/coefficients.txt", coefs)
    coefs = read_coefficients("lab8/coefficients.txt")
    print(f"Коефіцієнти збережено та зчитано: {coefs}")

    test_pts = [0.0, 1.0, 2.0, 3.0]
    print("\nПеревірка poly_eval:")
    for tp in test_pts:
        print(f"  P({tp}) = {poly_eval(coefs, tp):.4f}")

    x_plot = np.linspace(-3, 4, 500)
    y_plot = np.array([poly_eval(coefs, xi) for xi in x_plot])
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, y_plot, "b-", lw=2, label=r"$P(x) = x^3 - 4x^2 + 9x - 10$")
    plt.axhline(y=0, color="k", lw=0.8)
    plt.plot(2.0, 0, "ro", markersize=10, label="Дійсний корінь x = 2")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("Алгебраїчне рівняння P(x) = 0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab8/fig_polynomial.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    x0_real = 3.0
    root_real, it_nh = newton_horner(coefs, x0_real, eps, max_iter)
    print(f"\nМетод Ньютона зі схемою Горнера (x0 = {x0_real}):")
    print(f"  Дійсний корінь: x = {root_real:.10f}")
    print(f"  Ітерацій: {it_nh}")
    print(f"  P(x) = {poly_eval(coefs, root_real):.4e}")

    alpha0, beta0 = 0.8, 1.8
    alpha, beta, it_lin = lin_method(coefs, alpha0, beta0, eps, 100)
    print(f"\nМетод Ліна (alpha0 = {alpha0}, beta0 = {beta0}):")
    print(f"  Комплексні корені: x = {alpha:.10f} +/- {beta:.10f}i")
    print(f"  Ітерацій: {it_lin}")
    z = complex(alpha, beta)
    pz = poly_eval(coefs, z)
    print(f"  |P(alpha + beta*i)| = {abs(pz):.4e}")

    print("\nФайли: lab8/tabulation.txt, lab8/coefficients.txt, lab8/fig_*.png")

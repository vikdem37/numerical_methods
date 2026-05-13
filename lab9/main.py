import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return x1**2 + x2**2 - 4


def f2(x1, x2):
    return x2 - x1**2 + 1


def phi(X):
    return f1(X[0], X[1])**2 + f2(X[0], X[1])**2


def rosenbrock(X):
    return 100 * (X[0]**2 - X[1])**2 + (X[0] - 1)**2


def exploratory_search(func, x_base, dx, q, eps1, reduce_step):
    n = len(x_base)
    x_new = x_base.copy()
    dx = dx.copy()

    for i in range(n):
        while True:
            x_try = x_new.copy()
            x_try[i] = x_base[i] + dx[i]
            if func(x_try) < func(x_new):
                x_new = x_try
                break

            x_try[i] = x_base[i] - dx[i]
            if func(x_try) < func(x_new):
                x_new = x_try
                break

            if not reduce_step:
                break

            dx[i] /= q
            if dx[i] < eps1:
                break

    return x_new, dx


def hooke_jeeves(func, x0, dx0, q, p, eps1, eps2, max_iter):
    x0 = np.array(x0, dtype=float)
    dx_init = np.array(dx0, dtype=float)
    trajectory = [x0.copy()]

    for iteration in range(1, max_iter + 1):
        dx = dx_init.copy()
        x1, dx = exploratory_search(func, x0, dx, q, eps1, reduce_step=True)
        trajectory.append(x1.copy())

        if np.allclose(x1, x0):
            return x0, func(x0), iteration, trajectory

        diff_x = np.max(np.abs(x1 - x0))
        diff_f = abs(func(x1) - func(x0))
        if diff_x < eps1 and diff_f < eps2:
            return x1, func(x1), iteration, trajectory

        while True:
            x_p = x1 + p * (x1 - x0)
            x2, _ = exploratory_search(func, x_p, dx_init.copy(), q, eps1, reduce_step=False)
            trajectory.append(x2.copy())

            if func(x2) < func(x1):
                x0 = x1.copy()
                x1 = x2.copy()

                diff_x = np.max(np.abs(x1 - x0))
                diff_f = abs(func(x1) - func(x0))
                if diff_x < eps1 and diff_f < eps2:
                    return x1, func(x1), iteration, trajectory
            else:
                x0 = x1.copy()
                break

    return x0, func(x0), max_iter, trajectory


def save_trajectory(path, trajectory, func):
    with open(path, "w", encoding="utf-8") as f:
        n = len(trajectory[0])
        header = "  ".join(f"{'x' + str(i+1):>14}" for i in range(n))
        f.write(f"{'Крок':>6}  {header}  {'f(X)':>14}\n")
        for k, pt in enumerate(trajectory):
            vals = "  ".join(f"{pt[i]:14.8f}" for i in range(n))
            f.write(f"{k:6d}  {vals}  {func(pt):14.8e}\n")


if __name__ == "__main__":
    q = 2
    p = 2
    eps1 = 1e-8
    eps2 = 1e-10
    max_iter = 10000

    print("Тестування на функцiї Розенброка")
    print("f(X) = 100*(x1^2 - x2)^2 + (x1 - 1)^2")
    print("Точний мiнiмум: (1, 1), f = 0")
    print(f"X(0) = (-1.2, 0.0), dx = (0.5, 0.5), q = {q}, p = {p}")

    x_r, f_r, it_r, traj_r = hooke_jeeves(
        rosenbrock, [-1.2, 0.0], [0.5, 0.5], q, p, eps1, eps2, max_iter
    )

    print(f"\nРезультат: x = ({x_r[0]:.10f}, {x_r[1]:.10f})")
    print(f"f(x) = {f_r:.4e}")
    print(f"Iтерацiй: {it_r}")
    print(f"Точок траєкторiї: {len(traj_r)}")
    print(f"Похибка: |x1 - 1| = {abs(x_r[0] - 1):.4e}, |x2 - 1| = {abs(x_r[1] - 1):.4e}")

    print("Система нелiнiйних рiвнянь (m = 2):")
    print("  f1(x1, x2) = x1^2 + x2^2 - 4 = 0  (коло)")
    print("  f2(x1, x2) = x2 - x1^2 + 1 = 0    (парабола)")

    x1_exact = np.sqrt((1 + np.sqrt(13)) / 2)
    x2_exact = (-1 + np.sqrt(13)) / 2
    print(f"\nТочнi розв'язки: ({x1_exact:.10f}, {x2_exact:.10f})")
    print(f"                 ({-x1_exact:.10f}, {x2_exact:.10f})")

    theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = 2 * np.cos(theta)
    circle_y = 2 * np.sin(theta)
    par_x = np.linspace(-2.5, 2.5, 500)
    par_y = par_x**2 - 1

    plt.figure(figsize=(10, 8))
    plt.plot(circle_x, circle_y, "b-", lw=2, label=r"$x_1^2 + x_2^2 = 4$")
    plt.plot(par_x, par_y, "r-", lw=2, label=r"$x_2 = x_1^2 - 1$")
    plt.plot([x1_exact, -x1_exact], [x2_exact, x2_exact], "ko",
             markersize=10, label="Розв'язки")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Система нелiнiйних рiвнянь")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("lab9/fig_system.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nМетод Хука-Дживса для розв'язання системи")
    print(f"X(0) = (1.5, 1.5), dx = (0.5, 0.5), q = {q}, p = {p}")
    print(f"eps1 = {eps1:.0e}, eps2 = {eps2:.0e}")

    x_sol, f_sol, it_sol, traj_sol = hooke_jeeves(
        phi, [1.5, 1.5], [0.5, 0.5], q, p, eps1, 1e-12, max_iter
    )

    print(f"\nРозв'язок: x = ({x_sol[0]:.10f}, {x_sol[1]:.10f})")
    print(f"phi(x) = {f_sol:.4e}")
    print(f"f1(x) = {f1(x_sol[0], x_sol[1]):.4e}")
    print(f"f2(x) = {f2(x_sol[0], x_sol[1]):.4e}")
    print(f"Iтерацiй: {it_sol}")
    print(f"Точок траєкторiї: {len(traj_sol)}")
    print(f"Похибка: |x1 - x1*| = {abs(x_sol[0] - x1_exact):.4e}, "
          f"|x2 - x2*| = {abs(x_sol[1] - x2_exact):.4e}")

    save_trajectory("lab9/trajectory.txt", traj_sol, phi)
    print(f"\nТраєкторiю збережено у lab9/trajectory.txt ({len(traj_sol)} точок)")

    traj_arr = np.array(traj_sol)
    x1_min, x1_max = traj_arr[:, 0].min() - 0.5, traj_arr[:, 0].max() + 0.5
    x2_min, x2_max = traj_arr[:, 1].min() - 0.5, traj_arr[:, 1].max() + 0.5

    x1_grid = np.linspace(x1_min, x1_max, 300)
    x2_grid = np.linspace(x2_min, x2_max, 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = phi(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(10, 8))
    levels = np.logspace(-4, 2, 30)
    plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.6)
    plt.colorbar(label="phi(X)")
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], "r.-", markersize=4, lw=0.8,
             label="Траєкторiя")
    plt.plot(traj_arr[0, 0], traj_arr[0, 1], "gs", markersize=10, label="Початок")
    plt.plot(traj_arr[-1, 0], traj_arr[-1, 1], "r*", markersize=14, label="Результат")
    plt.plot(circle_x, circle_y, "b--", lw=1, alpha=0.5)
    plt.plot(par_x, par_y, "r--", lw=1, alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Метод Хука-Дживса: контури phi(X) та траєкторiя")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab9/fig_optimization.png", dpi=150)
    if plt.get_backend() != "agg":
        plt.show()
    plt.close()

    print("\nФайли: lab9/trajectory.txt, lab9/fig_system.png, lab9/fig_optimization.png")

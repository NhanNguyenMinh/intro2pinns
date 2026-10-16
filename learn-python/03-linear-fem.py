"""Bài 03: giải hệ tuyến tính và FEM Hermite cho dầm tựa đơn.

Chạy trong CMD: python 03-linear-fem.py
Nghiệm được đánh giá bên trong phần tử, không chỉ tại các nút.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import solve


def element_matrices(h: float, bending_stiffness: float = 1.0, load: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ma trận phần tử với thứ tự [w_i, theta_i, w_j, theta_j], theta=w'."""
    k = (bending_stiffness / h**3) * np.array(
        [
            [12, 6 * h, -12, 6 * h],
            [6 * h, 4 * h**2, -6 * h, 2 * h**2],
            [-12, -6 * h, 12, -6 * h],
            [6 * h, 2 * h**2, -6 * h, 4 * h**2],
        ],
        dtype=float,
    )
    f = (load * h / 12) * np.array([6, h, 6, -h], dtype=float)
    return k, f


def assemble_beam(elements: int, length: float = 1.0, bending_stiffness: float = 1.0, load: float = 1.0) -> np.ndarray:
    """Lắp ráp và giải hệ; trả về [w_0,theta_0,...,w_n,theta_n]."""
    if elements < 1 or length <= 0 or bending_stiffness <= 0:
        raise ValueError("elements >= 1, length > 0 and EI > 0 are required")
    h = length / elements
    ndof = 2 * (elements + 1)
    stiffness = np.zeros((ndof, ndof))
    force = np.zeros(ndof)
    k_e, f_e = element_matrices(h, bending_stiffness, load)

    for element in range(elements):
        dofs = np.arange(2 * element, 2 * element + 4)
        stiffness[np.ix_(dofs, dofs)] += k_e
        force[dofs] += f_e

    # Dầm tựa đơn: chỉ w ở hai đầu bằng 0; góc xoay là bậc tự do tự do.
    fixed = np.array([0, 2 * elements])
    free = np.setdiff1d(np.arange(ndof), fixed)
    solution = np.zeros(ndof)
    solution[free] = solve(stiffness[np.ix_(free, free)], force[free], assume_a="pos")
    assert np.allclose(stiffness, stiffness.T)
    return solution


def evaluate_beam(points: np.ndarray, solution: np.ndarray, length: float = 1.0, bending_stiffness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy w_h và M_h=-EI*w_h'' tại các điểm trong [0,L]."""
    elements = len(solution) // 2 - 1
    h = length / elements
    points = np.asarray(points, dtype=float)
    if np.any((points < 0) | (points > length)):
        raise ValueError("Evaluation points must be in [0,L]")
    element_ids = np.minimum((points / h).astype(int), elements - 1)
    s = (points - element_ids * h) / h
    n = np.stack(
        (1 - 3 * s**2 + 2 * s**3, h * (s - 2 * s**2 + s**3),
         3 * s**2 - 2 * s**3, h * (-s**2 + s**3)),
        axis=1,
    )
    n_xx = np.stack(
        ((-6 + 12 * s) / h**2, (-4 + 6 * s) / h,
         (6 - 12 * s) / h**2, (-2 + 6 * s) / h),
        axis=1,
    )
    dofs = 2 * element_ids[:, None] + np.arange(4)
    local_solution = solution[dofs]
    displacement = np.sum(n * local_solution, axis=1)
    moment = -bending_stiffness * np.sum(n_xx * local_solution, axis=1)
    return displacement, moment


def reference_fields(points: np.ndarray, length: float = 1.0, bending_stiffness: float = 1.0, load: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    w = load * (points**4 - 2 * length * points**3 + length**3 * points) / (24 * bending_stiffness)
    moment = load * points * (length - points) / 2
    return w, moment


def relative_l2(numerical: np.ndarray, reference: np.ndarray, points: np.ndarray) -> float:
    return float(np.sqrt(trapezoid((numerical - reference) ** 2, points) / trapezoid(reference**2, points)))


def main() -> None:
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = solve(a, b)
    print("System A @ x = b: x =", x, "; residual =", np.linalg.norm(a @ x - b))
    print(f"{'elements':>8} {'error w':>14} {'error M':>14}")
    for elements in (1, 2, 4, 8):
        solution = assemble_beam(elements)
        # Tránh trùng nút FEM khi đánh giá mômen, vốn có thể nhận hai giá trị phía trái/phải.
        points = (np.arange(2000) + 0.5) / 2000
        w_h, m_h = evaluate_beam(points, solution)
        w_star, m_star = reference_fields(points)
        print(f"{elements:8d} {relative_l2(w_h,w_star,points):14.6e} {relative_l2(m_h,m_star,points):14.6e}")
    assert np.allclose(assemble_beam(2)[[0, 4]], 0)


if __name__ == "__main__":
    main()

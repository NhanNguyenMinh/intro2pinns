"""Bài 02: mảng NumPy, đồ thị Matplotlib và kiểm tra ký hiệu bằng SymPy.

Chạy trong CMD: python 02-numpy-plot-sympy.py
Hình được lưu trong outputs/beam_reference.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lưu hình được cả khi CMD không có cửa sổ đồ họa
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def reference_fields(x: np.ndarray, length: float = 1.0, bending_stiffness: float = 1.0, load: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Trả về chuyển vị w và mômen M=-EI*w'' trên một mảng điểm."""
    w = load * (x**4 - 2 * length * x**3 + length**3 * x) / (24 * bending_stiffness)
    moment = load * x * (length - x) / 2
    return w, moment


def verify_symbolically() -> None:
    x, length, stiffness, load = sp.symbols("x L EI q0", positive=True)
    w = load * (x**4 - 2 * length * x**3 + length**3 * x) / (24 * stiffness)
    residual = sp.simplify(stiffness * sp.diff(w, x, 4) - load)
    boundary_values = [
        sp.simplify(w.subs(x, 0)),
        sp.simplify(w.subs(x, length)),
        sp.simplify(sp.diff(w, x, 2).subs(x, 0)),
        sp.simplify(sp.diff(w, x, 2).subs(x, length)),
    ]
    assert residual == 0 and boundary_values == [0, 0, 0, 0]
    print("SymPy: equation and four boundary conditions passed.")


def plot_pinn_results() -> None:
    """Đọc CSV do bài 04 tạo và vẽ trong một tiến trình không import PyTorch."""
    output_dir = Path(__file__).resolve().parent / "outputs"
    curve_path = output_dir / "pinn_curve.csv"
    loss_path = output_dir / "pinn_loss.csv"
    if not curve_path.exists() or not loss_path.exists():
        raise FileNotFoundError("Run python 04-torch-pinn.py --steps 200 first")
    curve = np.genfromtxt(curve_path, delimiter=",", names=True)
    losses = np.genfromtxt(loss_path, delimiter=",", names=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    axes[0].plot(curve["xi"], curve["u_exact"], color="#102a3a", label="exact")
    axes[0].plot(curve["xi"], curve["u_pinn"], "--", color="#087e83", label="PINN")
    axes[0].set(xlabel="xi", ylabel="normalized displacement u")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(losses["step"], losses["loss"], color="#b77727")
    axes[1].set(xlabel="Adam step", ylabel="training loss")
    axes[1].grid(alpha=0.25)
    output = output_dir / "pinn_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved PINN figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy, Matplotlib and SymPy for the beam reference")
    parser.add_argument("--plot-pinn", action="store_true", help="plot CSV files created by lesson 04")
    args = parser.parse_args()
    verify_symbolically()
    x = np.linspace(0.0, 1.0, 201)
    w, moment = reference_fields(x)
    print(f"NumPy: shape(x)={x.shape}, max(w)={w.max():.8f}, max(M)={moment.max():.8f}")

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [1.0, 2.0]])
    print("a * b (elementwise) =\n", a * b)
    print("a @ b (matrix product) =\n", a @ b)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    axes[0].plot(x, w, color="#087e83", label="w chính xác")
    axes[0].set_ylabel("Chuyển vị w")
    axes[1].plot(x, moment, color="#b77727", label="M chính xác")
    axes[1].set(xlabel="Vị trí x/L", ylabel="Mômen M")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    output = Path(__file__).resolve().parent / "outputs" / "beam_reference.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved figure: {output}")
    if args.plot_pinn:
        plot_pinn_results()


if __name__ == "__main__":
    main()

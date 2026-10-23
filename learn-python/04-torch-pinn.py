"""Bài 04: đạo hàm tự động và một PINN tối thiểu cho dầm tựa đơn.

Chạy trong CMD: python 04-torch-pinn.py --steps 200
Đây là baseline học tập, không phải kết luận cuối về độ chính xác của PINN.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn


def derivative(output: torch.Tensor, coordinate: torch.Tensor, order: int) -> torch.Tensor:
    """Đạo hàm theo tọa độ; mỗi đầu ra chỉ phụ thuộc vào điểm đầu vào cùng hàng."""
    result = output
    for _ in range(order):
        # create_graph=True giữ đồ thị của đạo hàm để tính tiếp bậc cao và backprop theo trọng số.
        result = torch.autograd.grad(
            result,
            coordinate,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
        )[0]
    return result


def verify_autograd() -> None:
    x = torch.linspace(0, 1, 5, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    polynomial = x**4 - 2 * x**3 + x
    fourth = derivative(polynomial, x, 4)
    assert torch.allclose(fourth, torch.full_like(fourth, 24.0), atol=1e-12)
    print("Autograd: fourth derivative of x^4-2*x^3+x is 24 (passed).")


def make_model() -> nn.Module:
    """Tanh đủ trơn để residual dạng mạnh dùng đạo hàm bậc bốn."""
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).double()


def train_pinn(steps: int = 200, n_collocation: int = 32, boundary_weight: float = 10.0, seed: int = 7) -> tuple[nn.Module, list[float]]:
    if steps < 1 or n_collocation < 2 or boundary_weight <= 0:
        raise ValueError("steps >= 1, n_collocation >= 2 and boundary_weight > 0 are required")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # Điểm giữa các ô: nơi đo residual PDE; cần gradient theo x để lấy u''''.
    collocation = ((torch.arange(n_collocation, dtype=torch.float64) + 0.5) / n_collocation).reshape(-1, 1).requires_grad_(True)
    # Hai đầu dầm: cần gradient theo x để lấy u'' trong điều kiện biên.
    boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    history = []

    for _ in range(steps):
        # PyTorch cộng dồn gradient mặc định; xóa gradient trọng số từ step trước.
        optimizer.zero_grad()
        u_c = model(collocation)
        # Residual của u''''=1; bình phương rồi lấy trung bình trên điểm miền.
        residual = derivative(u_c, collocation, 4) - 1.0
        loss_pde = torch.mean(residual**2)

        u_b = model(boundary)
        # Điều kiện tựa đơn: u=0 và u''=0 ở cả hai đầu.
        u_b_xx = derivative(u_b, boundary, 2)
        loss_boundary = torch.sum(u_b**2 + u_b_xx**2)
        loss = loss_pde + boundary_weight * loss_boundary
        # Tính gradient của loss theo các trọng số mạng, rồi Adam cập nhật chúng.
        loss.backward()
        optimizer.step()
        # Lưu một số thực, không giữ đồ thị autograd của mọi step trong history.
        history.append(float(loss.detach()))

        # Tọa độ là lá của đồ thị autograd; không dùng gradient theo tọa độ để tối ưu.
        collocation.grad = None
        boundary.grad = None
    return model, history


def evaluate(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    # Lưới kiểm chứng độc lập; vẫn cần gradient theo x để tính mômen -u''.
    coordinate = torch.linspace(0, 1, 401, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    predicted = model(coordinate)
    predicted_moment = -derivative(predicted, coordinate, 2)
    exact = (coordinate**4 - 2 * coordinate**3 + coordinate) / 24
    exact_moment = coordinate * (1 - coordinate) / 2
    x = coordinate[:, 0]
    error_u = torch.sqrt(torch.trapz((predicted[:, 0] - exact[:, 0]) ** 2, x) / torch.trapz(exact[:, 0] ** 2, x))
    error_m = torch.sqrt(torch.trapz((predicted_moment[:, 0] - exact_moment[:, 0]) ** 2, x) / torch.trapz(exact_moment[:, 0] ** 2, x))
    return (coordinate.detach(), predicted.detach(), exact.detach(),
            predicted_moment.detach(), exact_moment.detach(),
            float(error_u.detach()), float(error_m.detach()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal PINN for normalized beam u''''=1")
    parser.add_argument("--steps", type=int, default=200, help="number of Adam steps")
    parser.add_argument("--seed", type=int, default=7, help="random seed for network initialization")
    parser.add_argument("--n-collocation", type=int, default=32, help="number of interior PDE points")
    parser.add_argument("--boundary-weight", type=float, default=10.0, help="weight of boundary loss")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="directory for CSV results")
    args = parser.parse_args()

    verify_autograd()
    started = time.perf_counter()
    model, history = train_pinn(steps=args.steps, n_collocation=args.n_collocation,
                                boundary_weight=args.boundary_weight, seed=args.seed)
    train_seconds = time.perf_counter() - started
    x, predicted, exact, predicted_moment, exact_moment, error_u, error_m = evaluate(model)
    print(f"PyTorch {torch.__version__}; seed={args.seed}; steps={args.steps}; "
          f"n_collocation={args.n_collocation}; boundary_weight={args.boundary_weight}")
    print(f"loss: {history[0]:.4e} -> {history[-1]:.4e}")
    print(f"Training time: {train_seconds:.3f} s")
    print(f"Relative errors: u={error_u:.4e}, M={error_m:.4e}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "pinn_curve.csv"
    loss_path = output_dir / "pinn_loss.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["xi", "u_pinn", "u_exact", "M_pinn", "M_exact"])
        writer.writerows(zip(x[:, 0].tolist(), predicted[:, 0].tolist(),
                             exact[:, 0].tolist(), predicted_moment[:, 0].tolist(),
                             exact_moment[:, 0].tolist()))
    with loss_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows((step, value) for step, value in enumerate(history, start=1))
    print(f"Saved data: {curve_path} and {loss_path}")


if __name__ == "__main__":
    main()

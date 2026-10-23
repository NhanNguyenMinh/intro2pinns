"""PINN TensorFlow tối giản cho dầm tựa đơn: u''''=1 trên (0,1).

Chạy: python 04-tensorflow-pinn.py --steps 300
Đây là ví dụ học tập tương ứng với bản PyTorch, không phải mã gốc Katsikis.
"""

import argparse
import csv
import time
from pathlib import Path

import tensorflow as tf


def beam_derivatives(model, x):
    """Trả u, u'' và u''''; các hàng đầu vào độc lập với nhau."""
    with tf.GradientTape() as tape4:
        tape4.watch(x)
        with tf.GradientTape() as tape3:
            tape3.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape() as tape1:
                    tape1.watch(x)
                    u = model(x)
                u1 = tape1.gradient(u, x)
            u2 = tape2.gradient(u1, x)
        u3 = tape3.gradient(u2, x)
    u4 = tape4.gradient(u3, x)
    return u, u2, u4


def make_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(1),
    ])


def train_pinn(steps=300, n_collocation=32, boundary_weight=10.0, seed=7):
    if steps < 1 or n_collocation < 2 or boundary_weight <= 0:
        raise ValueError("steps >= 1, n_collocation >= 2, boundary_weight > 0")
    tf.keras.utils.set_random_seed(seed)
    model = make_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    indices = tf.cast(tf.range(n_collocation), tf.float64)
    x_c = tf.reshape((indices + 0.5) / n_collocation, (-1, 1))
    x_b = tf.constant([[0.0], [1.0]], dtype=tf.float64)
    history = []

    for _ in range(steps):
        with tf.GradientTape() as tape_theta:
            _, _, u4_c = beam_derivatives(model, x_c)
            u_b, u2_b, _ = beam_derivatives(model, x_b)
            loss_pde = tf.reduce_mean((u4_c - 1.0) ** 2)
            loss_bc = tf.reduce_sum(u_b ** 2 + u2_b ** 2)
            loss = loss_pde + boundary_weight * loss_bc
        weights = model.trainable_variables
        gradients = tape_theta.gradient(loss, weights)
        optimizer.apply_gradients(zip(gradients, weights))
        history.append(float(loss.numpy()))
    return model, history


def evaluate(model):
    x = tf.reshape(tf.linspace(tf.constant(0.0, tf.float64),
                               tf.constant(1.0, tf.float64), 401), (-1, 1))
    predicted, second, _ = beam_derivatives(model, x)
    moment = -second
    exact = (x ** 4 - 2 * x ** 3 + x) / 24.0
    exact_moment = x * (1.0 - x) / 2.0
    error_u = tf.sqrt(tf.reduce_sum((predicted - exact) ** 2) /
                      tf.reduce_sum(exact ** 2))
    error_m = tf.sqrt(tf.reduce_sum((moment - exact_moment) ** 2) /
                      tf.reduce_sum(exact_moment ** 2))
    return x, predicted, exact, moment, exact_moment, float(error_u), float(error_m)


def main():
    parser = argparse.ArgumentParser(description="TensorFlow PINN: u''''=1")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-collocation", type=int, default=32)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs-tensorflow")
    args = parser.parse_args()

    tf.keras.backend.set_floatx("float64")  # gọi trước khi tạo các lớp Dense
    started = time.perf_counter()
    model, history = train_pinn(args.steps, args.n_collocation,
                                args.boundary_weight, args.seed)
    train_seconds = time.perf_counter() - started
    x, predicted, exact, moment, exact_moment, error_u, error_m = evaluate(model)
    print(f"TensorFlow {tf.__version__}; seed={args.seed}; steps={args.steps}; "
          f"n_collocation={args.n_collocation}; boundary_weight={args.boundary_weight}")
    print(f"loss: {history[0]:.4e} -> {history[-1]:.4e}")
    print(f"Training time: {train_seconds:.3f} s")
    print(f"Relative errors: u={error_u:.4e}, M={error_m:.4e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "pinn_curve.csv"
    loss_path = args.output_dir / "pinn_loss.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["xi", "u_pinn", "u_exact", "M_pinn", "M_exact"])
        writer.writerows(zip(x[:, 0].numpy(), predicted[:, 0].numpy(),
                             exact[:, 0].numpy(), moment[:, 0].numpy(),
                             exact_moment[:, 0].numpy()))
    with loss_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows(enumerate(history, start=1))
    print(f"Saved data: {curve_path} and {loss_path}")


if __name__ == "__main__":
    main()

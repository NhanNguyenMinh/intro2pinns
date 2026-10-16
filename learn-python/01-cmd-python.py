"""Bài 01: Python cơ bản qua nghiệm giải tích của dầm tựa đơn.

Chạy trong CMD: python 01-cmd-python.py
Chỉ dùng thư viện chuẩn của Python.
"""


def beam_deflection(x: float, length: float = 1.0, bending_stiffness: float = 1.0, load: float = 1.0) -> float:
    """Chuyển vị w(x), dương theo chiều tải phân bố đều."""
    if length <= 0 or bending_stiffness <= 0 or load <= 0:
        raise ValueError("length, bending_stiffness, load must be positive")
    if not 0 <= x <= length:
        raise ValueError("x must be in [0, length]")
    return load * (x**4 - 2 * length * x**3 + length**3 * x) / (24 * bending_stiffness)


def main() -> None:
    parameters = {"length": 1.0, "bending_stiffness": 1.0, "load": 1.0}
    print("Lesson 01 - exact Euler-Bernoulli beam solution")
    print(f"Parameters: {parameters}")
    print(f"{'x':>8} {'w(x)':>14} {'location':>14}")

    for index in range(11):
        x = parameters["length"] * index / 10
        w = beam_deflection(x, **parameters)
        location = "midspan" if index == 5 else "support" if index in (0, 10) else ""
        print(f"{x:8.3f} {w:14.8f} {location:>14}")

    midpoint = parameters["length"] / 2
    expected = 5 * parameters["load"] * parameters["length"] ** 4 / (
        384 * parameters["bending_stiffness"]
    )
    assert abs(beam_deflection(0, **parameters)) < 1e-14
    assert abs(beam_deflection(parameters["length"], **parameters)) < 1e-14
    assert abs(beam_deflection(midpoint, **parameters) - expected) < 1e-14
    print(f"Midspan check: w(L/2) = {expected:.8f} (passed)")


if __name__ == "__main__":
    main()

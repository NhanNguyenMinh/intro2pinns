# Python cho dự án intro2pinns

Bốn bài thực hành dành cho sinh viên đã học C nhưng chưa dùng Python. Mỗi bài có một trang hướng dẫn HTML, một notebook để thử từng ô và một script Python chạy từ CMD. Tất cả dùng bài toán dầm Euler–Bernoulli tựa đơn với \(L=EI=q_0=1\), trừ khi ghi khác.

Có thể học trực tiếp trên trình duyệt: [hướng dẫn Google Colab và làm việc nhóm](colab.html) giải thích cách mở notebook, chia sẻ bản trên Drive và chốt phiên bản đã kiểm tra trên GitHub.

| Bài | Đọc trên website | Notebook | Script |
| --- | --- | --- | --- |
| 01. CMD và Python cơ bản | [Trang 01](01-cmd-python.html) | [Notebook 01](01-cmd-python.ipynb) | [Script 01](01-cmd-python.py) |
| 02. NumPy, đồ thị, SymPy | [Trang 02](02-numpy-plot-sympy.html) | [Notebook 02](02-numpy-plot-sympy.ipynb) | [Script 02](02-numpy-plot-sympy.py) |
| 03. Hệ tuyến tính và FEM | [Trang 03](03-linear-fem.html) | [Notebook 03](03-linear-fem.ipynb) | [Script 03](03-linear-fem.py) |
| 04. PyTorch và PINN đầu tiên | [Trang 04](04-torch-pinn.html) | [Notebook 04 tự chứa](04-torch-pinn.ipynb) | [Script 04](04-torch-pinn.py) |
| 05. Tích hợp FEM–PINN | [Mở trên Colab](https://colab.research.google.com/github/NhanNguyenMinh/intro2pinns/blob/main/learn-python/05-beam-fem-to-pinn.ipynb) | [Notebook 05](05-beam-fem-to-pinn.ipynb) | — |

## Cài đặt trên Windows

1. Cài [Anaconda Distribution hoặc Miniconda](https://www.anaconda.com/download). Mở **Anaconda Prompt**; đây là Command Prompt đã nhận diện lệnh conda. Nếu dùng CMD thông thường, hãy cấu hình conda theo [hướng dẫn chính thức](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) và mở lại CMD.
2. Trong thư mục learn-python, chạy:

        conda env create -f environment.yml
        conda activate intro2pinns-learning
        python --version
        where python

   Dòng đầu của where python cần trỏ tới môi trường intro2pinns-learning. Không cài thư viện bài học vào môi trường base.
3. Bài 04 cần PyTorch. Trên [trang cài đặt chính thức](https://pytorch.org/get-started/locally/), chọn **Stable → Windows → Pip → Python → CPU**, rồi chạy lệnh mà trang cung cấp **sau khi đã kích hoạt môi trường**. CPU đủ cho bài toán dầm 1D khởi đầu. Kiểm tra:

        python -c "import torch; print(torch.__version__)"

## Cách học

Trong CMD, chuyển vào thư mục này rồi chạy:

    cd /d DUONG_DAN_DEN_REPO\learn-python
    conda activate intro2pinns-learning
    python 01-cmd-python.py
    python 02-numpy-plot-sympy.py
    python 03-linear-fem.py
    python 04-torch-pinn.py --steps 200
    python 02-numpy-plot-sympy.py --plot-pinn

Mở notebook bằng lệnh jupyter lab trong cùng môi trường, chọn kernel intro2pinns-learning nếu có nhiều kernel. Notebook phù hợp để thử từng ý; bài 03 và 04 đều tự chứa ví dụ ngắn để học trên Colab: hai vòng lặp lắp ghép FEM, một hàm autograd và một vòng train PINN. Script là bản có thể chạy lại từ đầu đến cuối. Không nộp notebook chỉ chạy được nhờ các ô đã thực hiện theo thứ tự khác.

Mỗi bài kết thúc bằng một minh chứng ngắn: câu trả lời cho câu hỏi trên trang HTML, mã đã chạy, và hình/bảng kết quả. Các hình do script tạo nằm trong thư mục outputs (không theo dõi bằng Git). Cả nhóm ghi phiên bản Python và thư viện khi báo cáo kết quả; file environment.yml là điểm bắt đầu để tái lập.

## Lệnh CMD tối thiểu

| Lệnh | Ý nghĩa |
| --- | --- |
| cd /d PATH | Chuyển thư mục, kể cả đổi ổ đĩa |
| dir | Xem file và thư mục hiện tại |
| python --version / where python | Kiểm tra Python đang dùng |
| conda activate intro2pinns-learning | Kích hoạt môi trường dự án |
| python ten_file.py | Chạy script |
| jupyter lab | Mở môi trường notebook |
| Ctrl+C | Dừng lệnh đang chạy trong CMD |

Tài liệu nền: [Python Tutorial](https://docs.python.org/3/tutorial/), [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html), [SciPy linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html), [PyTorch tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).

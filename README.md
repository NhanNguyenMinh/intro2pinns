# Intro to PINNs for Mechanics

**Dự án nghiên cứu sinh viên — TS. Nguyễn Minh Nhân**

Repo này là nơi nhóm ba sinh viên cùng học, cài đặt và đánh giá *physics-informed neural networks* (PINNs) qua một bài toán cơ học có nghiệm giải tích. Câu hỏi khởi đầu là:

> **Loss huấn luyện nhỏ có bảo đảm nghiệm PINN chính xác không?**

Nhóm sẽ dùng nghiệm giải tích làm chuẩn và phương pháp phần tử hữu hạn (FEM) làm mốc so sánh. Trọng tâm là hiểu mô hình toán, tạo thí nghiệm có thể tái lập và giải thích kết quả bằng sai số đo được.

**Mới bắt đầu với Python?** Học theo [bốn bài thực hành trong `learn-python/`](learn-python/README.md): CMD và Python cơ bản → NumPy/đồ thị/SymPy → hệ tuyến tính/FEM → PyTorch/PINN. Mỗi bài có [trang hướng dẫn](learn-python/index.html), notebook và script chạy độc lập.

## 1. Bài toán chuẩn: dầm Euler–Bernoulli 1D

Xét dầm tựa đơn dài \(L\), độ cứng uốn \(EI\) không đổi, chịu tải phân bố đều \(q_0\). Gọi \(w(x)\) là chuyển vị theo chiều tải:

$$
EI\,w^{(4)}(x)=q_0,\qquad 0<x<L,
$$
$$
w(0)=w(L)=0,\qquad w''(0)=w''(L)=0.
$$

Hai điều kiện \(w=0\) là chuyển vị tại gối tựa; hai điều kiện \(w''=0\) tương ứng với mômen uốn bằng không ở hai đầu. Chọn \(EI,L,q_0>0\) và quy ước chuyển vị cùng chiều tải là dương. Nghiệm giải tích của **mô hình dầm này** là

$$
w_*(x)=\frac{q_0}{24EI}
\left(x^4-2Lx^3+L^3x\right),\qquad
M_*(x)=-EI\,w_*''(x).
$$

Trước khi huấn luyện mạng, nhóm cần tự suy ra nghiệm và kiểm tra bằng cách thay vào phương trình cùng bốn điều kiện biên.

## 2. Ba cách giải và vai trò của chúng

| Cách tiếp cận | Việc cần làm | Vai trò trong dự án |
| --- | --- | --- |
| Giải tích | Tích phân ODE và áp điều kiện biên | Tạo nghiệm chuẩn \(w_*,M_*\) |
| FEM | Giải dạng yếu bằng phần tử Hermite bậc ba | Mốc so sánh về sai số và chi phí |
| PINN | Xấp xỉ \(w\) bằng mạng neural; tối thiểu hóa sai số phương trình và điều kiện biên | Phương pháp cần khảo sát |

Dạng yếu cho FEM: đặt \(V=\{v\in H^2(0,L):v(0)=v(L)=0\}\); tìm \(w\in V\) sao cho

$$
\int_0^L EI\,w''v''\,dx=\int_0^Lq_0v\,dx
\quad\text{với mọi }v\in V.
$$

Trong bài toán này, chuyển vị bằng không tại hai đầu được áp trực tiếp; mômen bằng không là điều kiện biên tự nhiên của dạng yếu.

Để huấn luyện PINN ở thang số thuận tiện, đặt

$$
\xi=\frac{x}{L},\qquad
u(\xi)=\frac{EI}{q_0L^4}w(x).
$$

Khi đó \(u^{(4)}=1\) trên \((0,1)\), còn \(u(0)=u(1)=u''(0)=u''(1)=0\). Với mạng \(u_\alpha\), một hàm loss cơ bản là

$$
\mathcal L(\alpha)=
\underbrace{\frac{1}{N_c}\sum_{i=1}^{N_c}
\bigl[u_\alpha^{(4)}(\xi_i)-1\bigr]^2}_{\mathcal L_{\mathrm{PDE}}}
+
\lambda_b\underbrace{\sum_{s\in\{0,1\}}
\bigl[u_\alpha(s)^2+u_\alpha''(s)^2\bigr]}_{\mathcal L_{\mathrm{BC}}}.
$$

Cần dùng hàm kích hoạt đủ trơn, chẳng hạn \(\tanh\), để tính đạo hàm bậc bốn bằng đạo hàm tự động. Giá trị loss tại các điểm huấn luyện **không thay thế** phép đo sai số nghiệm trên tập điểm độc lập.

## 3. Thí nghiệm đầu tiên

1. Tạo một baseline giải tích, một baseline FEM Hermite và một baseline PINN cho **cùng** \(EI,L,q_0\) và điều kiện biên.
2. Chọn **một** yếu tố để khảo sát: số điểm collocation \(N_c\) **hoặc** trọng số biên \(\lambda_b\). Giữ các yếu tố còn lại cố định, gồm kiến trúc mạng, optimizer, số bước huấn luyện, độ chính xác số và quy tắc lấy mẫu.
3. Chạy ít nhất ba seed khởi tạo cho mỗi cấu hình PINN; lưu seed và cấu hình cùng kết quả.
4. Tính sai số trên một lưới đánh giá cố định, độc lập với điểm collocation và nút FEM. Với trường FEM, đánh giá cả bên trong phần tử.
5. Vẽ sai số theo chi phí tính toán và kiểm tra xem loss nhỏ hơn có đi cùng sai số chuyển vị, mômen nhỏ hơn hay không.

Hai đại lượng chính:

$$
e_w=\frac{\|w_{\mathrm{num}}-w_*\|_{L^2(0,L)}}
{\|w_*\|_{L^2(0,L)}},\qquad
e_M=\frac{\|M_{\mathrm{num}}-M_*\|_{L^2(0,L)}}
{\|M_*\|_{L^2(0,L)}}.
$$

Ngoài ra, báo cáo sai số điều kiện biên, \(\mathcal L_{\mathrm{PDE}}\), \(\mathcal L_{\mathrm{BC}}\), thời gian lắp ráp và giải FEM, thời gian huấn luyện PINN, cùng thời gian dự đoán sau huấn luyện. Ghi nhận cả những lần chạy không hội tụ. So sánh thời gian trên cùng phần cứng và ghi rõ điều kiện dừng; nếu khả thi, so thời gian đạt cùng một mức sai số.

## 4. Cách ba thành viên phối hợp

Ba mảng việc khởi đầu là **mô hình và nghiệm giải tích**, **FEM và bộ đo sai số**, **PINN và thí nghiệm huấn luyện**. Nhóm tự thống nhất người phụ trách từng mảng; mỗi kết quả cần được ít nhất một thành viên khác kiểm tra. Người phụ trách PINN và FEM cùng thống nhất tham số vật lý, quy ước dấu và tập điểm đánh giá trước khi so sánh.

Mỗi thay đổi nên đi qua một nhánh riêng và pull request ngắn, nêu câu hỏi, cách kiểm tra và kết quả. Ghi quyết định thí nghiệm và các kết quả bất thường để cả nhóm có thể tái lập.

### Sản phẩm của giai đoạn khởi động

- Mã nguồn chạy được, có hướng dẫn cài đặt và chạy đúng với các file **thực sự có trong repo**.
- Nghiệm giải tích được kiểm tra; FEM và PINN cùng giải một bài toán.
- Bảng tham số, seed, sai số, thời gian; đồ thị nghiệm, loss và sai số theo chi phí.
- Báo cáo ngắn: đã kiểm chứng điều gì, PINN gặp hạn chế nào, và một hướng cải thiện có căn cứ.

## 5. Tài liệu nên đọc

**Đọc trước**

1. D. Katsikis, A. D. Muradova & G. E. Stavroulakis (2022), [*A Gentle Introduction to Physics-Informed Neural Networks, with Applications in Static Rod and Beam Problems*](https://doi.org/10.15377/2409-5761.2022.09.8). Đọc mục 3 và Problem 2 ở mục 4.1 để hiểu cách xây dựng loss, đạo hàm tự động và code cho dầm tựa đơn. **Lưu ý:** trang 117 gọi nhầm các điều kiện \(w=w''=0\) ở hai đầu là “totally clamped”; đó là điều kiện tựa đơn.
2. A. J. M. Ferreira & N. Fantuzzi (2020), [*MATLAB Codes for Finite Element Analysis: Solids and Structures*](https://doi.org/10.1007/978-3-030-47952-7), chương *Bernoulli Beams*, tr. 89–104. Dùng để xây dựng FEM Hermite đối chứng.

**Đọc tiếp khi đã có baseline**

3. M. Raissi, P. Perdikaris & G. E. Karniadakis (2019), [*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*](https://doi.org/10.1016/j.jcp.2018.10.045). Nền tảng phương pháp PINNs.
4. T. G. Grossmann, U. J. Komorowska, J. Latz & C.-B. Schönlieb (2024), [*Can physics-informed neural networks beat the finite element method?*](https://doi.org/10.1093/imamat/hxae011). Đọc mục 2–3 và 7 để thiết kế phép so sánh sai số–chi phí; bài này khảo sát các PDE khác, không phải dầm của dự án.

**Người hướng dẫn:** TS. Nguyễn Minh Nhân, Bộ môn Toán học tính toán, Khoa Toán–Tin học, Trường Đại học Khoa học tự nhiên, ĐHQG-HCM.

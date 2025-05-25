# Thuật toán Tìm kiếm cục bộ cho Bài toán Người bán hàng du lịch (TSP)

Dự án triển khai ba thuật toán tìm kiếm cục bộ hiệu quả để giải quyết bài toán TSP (Travelling Salesman Problem), bao gồm:

- **Tabu Search (TS)**
- **List-Based Simulated Annealing (LBSA)**
- **Guided Local Search (GLS)**

Mục tiêu là tìm lời giải gần tối ưu trong thời gian hợp lý trên nhiều bộ dữ liệu chuẩn TSPLIB.

---

## Mục lục
- [Cài đặt](#cài-đặt)
- [Chi tiết thuật toán](#chi-tiết-thuật-toán)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Cấu trúc thư mục dự án](#cấu-trúc-thư-mục-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Liên hệ](#liên-hệ)
---

## Cài đặt
Yêu Cầu Hệ Thống
- Python ≥ 3.8
- Các thư viện cần cài đặt: numpy, matplotlib,tqdm, pandas
  
Clone repo và cài đặt các thư viện cần thiết:

```bash
git clone https://github.com/viet12012003/local_search_algorithms_for_the_TSP_problem
pip install -r requirements.txt
```

## Chi tiết thuật toán
Chi tiết thuật toán có trong file: Link drive

---

## Bộ dữ liệu

Để đánh giá hiệu quả thuật toán, dự án đã sử dụng 10 bộ dữ liệu chuẩn từ TSPLIB (nguồn: [TSPLIB GitHub](https://github.com/mastqe/tsplib)). Đây là các bài toán TSP đối xứng với đầu vào là tọa độ các thành phố.

Đồng thời, nhóm sử dụng kết quả **Best Known Solutions (BKS)** từ [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html) để so sánh và đánh giá sai số tương đối của từng thuật toán.

Danh sách 10 bộ dữ liệu bao gồm: `eil51`, `eil76`, `kroA100`, `ch150`, `kroA200`, `pr264`, `lin318`, `rd400`, `pr439`, `rat575` — tương ứng với số lượng thành phố tăng dần.

Dữ liệu đã được xử lý lưu tại file data.txt

> **Lưu ý:** Dữ liệu được tiền xử lý để chỉ giữ lại tọa độ (x, y) của các thành phố. Từ đó, xây dựng ma trận khoảng cách Euclid làm đầu vào cho các thuật toán.

| Bộ dữ liệu | Số thành phố (n) | Số cạnh (n*(n-1)/2) | BKS  |
|------------|------------------|----------------------|------|
| eil51      | 51               | 1,275                | 426  |
| eil76      | 76               | 2,850                | 538  |
| kroA100    | 100              | 4,950                | 21,282 |
| ch150      | 150              | 11,175               | 6,528 |
| kroA200    | 200              | 19,900               | 29,368 |
| pr264      | 264              | 34,716               | 49,135 |
| lin318     | 318              | 50,403               | 42,029 |
| rd400      | 400              | 79,800               | 15,281 |
| pr439      | 439              | 96,141               | 107,217 |
| rat575     | 575              | 165,025              | 6,773 |

---
## Cấu trúc thư mục dự án

```
│
├── GLS/ # Thư mục chứa code thuật toán Guided Local Search (GLS)
│ └── gls.py
│
├── LBSA/ # Thư mục chứa code thuật toán LBSA
│ └── lbsa.py
│ └── config.py  # Chứa bộ tham số được sử dụng trong thuật toán LBSA
│ └── model.py
│ └── data.in  # Dữ liệu để kiểm thử thuật toán LBSA
│
├── TS/ # Thư mục chứa code thuật toán Tabu Search
│ └── TabuSearch.py
│ └── test_tabu_search.py
│ └── TSP.py
│
│── data.txt # File dữ liệu đã được tiền xử lý
│── test.txt # File dữ liệu chỉ có 2 bộ dữ liệu để kiểm thử chương trình
├── main.py # File chính thực thi chương trình (chạy thuật toán)
├── requirements.txt # Danh sách thư viện cần cài đặt
└── README.md # Tài liệu hướng dẫn này
```

## Hướng dẫn sử dụng

Chạy chương trình
File main.py là điểm khởi đầu để thực thi thuật toán giải bài toán Người giao hàng (TSP) bằng một trong ba thuật toán: TS, LBSA, hoặc GLS.

Cách chọn thuật toán
Mở file main.py, chỉnh sửa biến method theo thuật toán mong muốn:

``` bash
# Lựa chọn thuật toán:
method = "TS"     # Tabu Search
# method = "LBSA"   # LBSA (List-Based Simulated Annealing)
# method = "GLS"    # Guided Local Search
```

Cần thay đổi đường dẫn file ở dòng sau để chạy chương trình:
``` bash
with open('your_path\local_search_algorithms_for_the_TSP_problem\\test.txt', 'r') as f:
```

**Tham số quan trọng**
- num_tests: Số lần chạy thử mỗi dataset để tính toán trung bình.

- SHOW_VISUAL = True: Nếu True, kết quả sẽ hiển thị bằng biểu đồ Matplotlib.

- Một số tham số khác như alpha, tb_size, M, K... có thể chỉnh trực tiếp trong phần mã của từng thuật toán để tùy biến thuật toán

**Kết quả đầu ra**
Với mỗi bộ dữ liệu, chương trình sẽ in ra:

- Best Cost: Chi phí tốt nhất tìm được sau num_tests lần chạy.

- Avg Cost ± Std: Giá trị trung bình và độ lệch chuẩn của chi phí.

- Avg Time ± Std: Thời gian chạy trung bình.

Nếu SHOW_VISUAL = True, chương trình sẽ vẽ:

- Đồ thị lộ trình tốt nhất.

- Đồ thị quá trình cải thiện nghiệm (nếu có).

---
## Liên hệ

Nếu có bất kỳ câu hỏi, góp ý hoặc muốn đóng góp vào dự án, vui lòng liên hệ qua email:

Nguyễn Như Yến Phương – nguyennhuyenphuong_t67@hus.edu.vn

Vương Sỹ Việt – vuongsyviet_t67@hus.edu.vn

Ngô Hải Yến – ngohaiyen_t67@hus.edu.vn

Rất hoan nghênh mọi ý kiến đóng góp nhằm cải thiện dự án!

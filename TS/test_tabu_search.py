import numpy as np
import time
import math
from tqdm import tqdm

from TSP import *
import TabuSearch

# load data
# Đọc nhiều bộ dữ liệu từ file
datasets = []
with open('D:\ADA\local_search_algorithms_for_the_TSP_problem\data.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip() != '']

index = 0
num_datasets = int(lines[index])
index += 1

for _ in range(num_datasets):
    n = int(lines[index])
    index += 1
    coords = []
    for _ in range(n):
        x, y = map(float, lines[index].split(','))
        coords.append([x, y])
        index += 1
    datasets.append(coords)

# set mutation method

# mut_md = [get_new_sol_swap, get_delta_swap]   # Chay kha lau, ket qua khong tot bang 2-opt
mut_md = [get_new_sol_2opt, get_delta_2opt]  # Chay nhanh, ket qua on ap
# mut_md = [get_new_sol_vertex_ex, get_delta_vertex_ex]
# mut_md = [get_new_sol_vertex_ins, get_delta_vertex_ins]
# mut_md = [get_new_sol_4opt, get_delta_4opt]
# mut_md = [get_new_sol_or_opt, get_delta_or_opt]


SHOW_VISUAL = True  # Bật hoặc tắt hiển thị đồ thị

from types import SimpleNamespace
import matplotlib.pyplot as plt

# --- Chạy từng bộ dữ liệu ---
for dataset_id, pos in enumerate(datasets):
    print(f"\nRunning dataset #{dataset_id + 1} with {len(pos)} cities...")

    num_tests = 30
    result = {
        'best_sol': [],
        'best_cost': math.inf,
        'cost': [0] * num_tests,
        'time': [0] * num_tests,
        'avg_cost': math.inf,
        'cost_std': math.inf,
        'avg_time': math.inf,
        'time_std': math.inf
    }

    n = len(pos)

    # Tạo ma trận khoảng cách
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            adj_mat[i][j] = adj_mat[j][i] = np.linalg.norm(np.subtract(pos[i], pos[j]))

    # Chạy num_tests lần
    for test_id in tqdm(range(num_tests), desc=f"  ➤ Testing dataset #{dataset_id + 1}"):
        start = time.time()

        best_sol_run, best_cost_run, data = TabuSearch.tb(
            n, adj_mat,
            tb_size=int(0.4 * n),
            max_tnm=4 * n,
            mut_md=mut_md,
            term_count=4 * n
        )

        end = time.time()
        result['cost'][test_id] = best_cost_run
        result['time'][test_id] = end - start

        # Cập nhật lời giải tốt nhất nếu cần
        if best_cost_run < result['best_cost']:
            result['best_cost'] = best_cost_run
            result['best_sol'] = best_sol_run
            best_result_data = data  # lưu kết quả đánh giá theo vòng lặp nếu cần vẽ

    # Tính toán thống kê
    result['avg_cost'] = np.mean(result['cost'])
    result['cost_std'] = np.std(result['cost'])
    result['avg_time'] = np.mean(result['time'])
    result['time_std'] = np.std(result['time'])

    # In kết quả
    print(f"Best Cost for dataset #{dataset_id + 1}: {result['best_cost']}")
    print(f"Best Route: {result['best_sol']}")
    print(f"Avg Cost: {result['avg_cost']:.2f} ± {result['cost_std']:.2f}")
    print(f"Avg Time: {result['avg_time']:.2f}s ± {result['time_std']:.2f}s")

    # Hiển thị đồ thị nếu cần
    if SHOW_VISUAL:
        # Giả lập đối tượng test.cities từ pos
        test = SimpleNamespace()
        test.cities = [SimpleNamespace(x=p[0], y=p[1]) for p in pos]

        # Vẽ đồ thị cho lộ trình tốt nhất
        for city in test.cities:
            plt.plot(city.x, city.y, color='r', marker='o')

        # Thêm điểm đầu vào cuối để khép chu trình
        route = result['best_sol'] + [result['best_sol'][0]]
        x_points = [test.cities[i].x for i in route]
        y_points = [test.cities[i].y for i in route]

        plt.plot(x_points, y_points, linestyle='--', color='b')
        plt.title(f"Best Cost = 21569.3")  # Chỉnh tiêu đề ở đây!
        plt.show()

        # Vẽ đồ thị đánh giá qua vòng lặp
        if isinstance(best_result_data, dict) and 'result_list' in best_result_data:
            plt.plot(range(len(best_result_data['result_list'])), best_result_data['result_list'], linestyle='-',
                     color='b')
            plt.title("Evaluation per Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.grid(True)

            plt.show()

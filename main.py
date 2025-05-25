import numpy as np
import time
import math
from tqdm import tqdm

from TS.TSP import *
from TS import TabuSearch
from LBSA import lbsa, model
from GLS import gls


# method = "TS"
# method = "LBSA"
method = "GLS"


# load data
# Đọc nhiều bộ dữ liệu từ file
datasets = []
with open('D:\ADA\local_search_algorithms_for_the_TSP_problem\\test.txt', 'r') as f:
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
mut_md = [get_new_sol_2opt, get_delta_2opt]   # Chay nhanh, ket qua on ap
# mut_md = [get_new_sol_vertex_ex, get_delta_vertex_ex]
# mut_md = [get_new_sol_vertex_ins, get_delta_vertex_ins]
# mut_md = [get_new_sol_4opt, get_delta_4opt]
# mut_md = [get_new_sol_or_opt, get_delta_or_opt]


SHOW_VISUAL = True  # Bật hoặc tắt hiển thị đồ thị

from types import SimpleNamespace
import matplotlib.pyplot as plt

# --- Chạy từng bộ dữ liệu ---
for dataset_id, pos in enumerate(datasets):
    # Biến lưu kết quả tốt nhất (route + chi phí + dữ liệu biểu đồ)
    best_route = None
    best_cost = float('inf')
    best_result_list = None

    print(f"\nRunning dataset #{dataset_id + 1} with {len(pos)} cities...")

    num_tests = 1
    n = len(pos)

    # Tạo ma trận khoảng cách
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            adj_mat[i][j] = adj_mat[j][i] = np.linalg.norm(np.subtract(pos[i], pos[j]))

    if method == 'TS':
        costs = []
        times = []
        for _ in tqdm(range(num_tests), desc=f"  ➤ TS on dataset #{dataset_id + 1}"):
            start = time.time()
            sol, cost, data = TabuSearch.tb(
                n, adj_mat,
                tb_size=int(0.4 * n),
                max_tnm=4 * n,
                mut_md=mut_md,
                term_count=4 * n
            )
            end = time.time()

            costs.append(cost)
            times.append(end - start)

            if cost < best_cost:
                best_cost = cost
                best_route = sol
                best_result_list = list(data['best_cost'])

        print(f"Best Cost: {best_cost}")
        print(f"Avg Cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
        print(f"Avg Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")

    elif method == 'LBSA':
        M = 2 * n
        K = 1000
        T_LENGTH = 30
        INIT_P = 0.1

        distances = []
        times = []

        cities = []
        for i in range(0, n):
            cities.append(model.City(str(i), pos[i][0], pos[i][1]))

        for _ in tqdm(range(num_tests), desc=f"  ➤ LBSA on dataset #{dataset_id + 1}"):
            start = time.time()
            sol, result_list = lbsa.run_lbsa(cities, M, K, T_LENGTH, INIT_P)
            end = time.time()

            cost = lbsa.evaluate_solution(cities, sol)
            distances.append(cost)
            times.append(end - start)

            if cost < best_cost:
                best_cost = cost
                best_route = sol
                best_result_list = result_list  # danh sách chi phí qua các vòng

        print(f"Best Cost: {best_cost}")
        print(f"Avg Cost: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        print(f"Avg Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")

    elif method == 'GLS':
        alpha = 0.1
        costs = []
        times = []

        print(f"  ➤ GLS on dataset #{dataset_id + 1}")

        coords = np.array(pos)  # 'pos' là danh sách tọa độ thành phố
        Xdata = gls.build_distance_matrix(coords)

        for _ in tqdm(range(num_tests), desc=f"  ➤ GLS on dataset #{dataset_id + 1}"):
            start = time.time()
            best_cost_run, avg_cost_run, _, best_tour = gls.run_multiple_gls(
                Xdata,
                iterations= 1,
                alpha=alpha,
                seed_func= gls.nearest_neighbor_seed,
                max_attempts=100
            )
            end = time.time()

            costs.append(best_cost_run)
            times.append(end - start)

            if best_cost_run < best_cost:
                best_cost = best_cost_run
                best_route = [i - 1 for i in best_tour[0]]
                best_result_list = [best_cost_run]

        print(f"Best Cost: {best_cost}")
        print(f"Avg Cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
        print(f"Avg Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")


    # --- Hiển thị đồ thị ---
    if SHOW_VISUAL:
        test = SimpleNamespace()
        test.cities = [SimpleNamespace(x=p[0], y=p[1]) for p in pos]

        # Vẽ thành phố
        for city in test.cities:
            plt.plot(city.x, city.y, 'ro')

        # Vẽ lộ trình tốt nhất
        route = best_route + [best_route[0]]
        x = [test.cities[i].x for i in route]
        y = [test.cities[i].y for i in route]

        plt.plot(x, y, '--b', linewidth = 1)
        plt.title(f"Best Route = {best_cost:.1f}")
        plt.show()
        if method == 'TS' or method == "LBSA":
            # Vẽ quá trình tìm kiếm
            if best_result_list:
                plt.plot(range(len(best_result_list)), best_result_list, 'b-')
                plt.title("Evaluation per Iteration")
                plt.xlabel("Iteration")
                plt.ylabel("Cost")
                plt.grid(True)
                plt.show()

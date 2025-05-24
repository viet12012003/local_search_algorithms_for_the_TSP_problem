# tb.py
import random
import math
from collections import deque
from TSP import get_cost

def tnm_selection(n, adj_mat, sol, max_tnm, mut_md, tb_size, tb_list, fq_dict, best_cost):
    """
    :param n: số lượng đỉnh
    :param adj_mat: ma trận chi phí (khoảng cách giữa các thành phố)
    :param sol: lời giải hiện tại, nơi các hàng xóm (neighbours) được chọn từ đây
    :param max_tnm: số lượng ứng viên được chọn trong lựa chọn (tournament selection)
    :param mut_md: [get_sol, get delta], phương pháp đột biến, ví dụ như hoán đổi, 2-opt
    :param tb_size: >=0, kích thước tối đa của danh sách Tabu (tb_list)
    :param tb_list: deque, danh sách Tabu, tổ chức theo kiểu FIFO (out <- [...] <- in)
    :param fq_dict: từ điển lưu tần suất của từng cặp đỉnh (chưa sử dụng)
    :param best_cost: chi phí của lời giải tốt nhất hiện tại
    """

    get_new_sol = mut_md[0]  # Hàm tạo lời giải mới
    get_delta = mut_md[1]    # Hàm tính toán sự thay đổi chi phí (delta) khi thực hiện một move

    cost = get_cost(n, adj_mat, sol)  # Tính chi phí hiện tại của lời giải

    best_delta_0 = math.inf  # Khởi tạo delta tốt nhất chưa bị cấm
    best_i_0 = best_j_0 = -1  # Chỉ số của cặp đỉnh tốt nhất chưa bị cấm

    best_delta_1 = math.inf  # Khởi tạo delta tốt nhất bị cấm
    best_i_1 = best_j_1 = -1  # Chỉ số của cặp đỉnh tốt nhất bị cấm

    # Lặp qua tối đa `max_tnm` lần để tìm move tốt nhất
    for _ in range(max_tnm):
        i, j = random.sample(range(n), 2)  # Chọn ngẫu nhiên hai chỉ số đỉnh
        i, j = (i, j) if i < j else (j, i)  # Đảm bảo i < j để tránh lặp lại (i, j) và (j, i)
        v_1, v_2 = (sol[i], sol[j]) if sol[i] < sol[j] else (sol[j], sol[i])  # Sắp xếp để tiện cho tra cứu trong danh sách Tabu
        delta = get_delta(n, adj_mat, sol, i, j)  # Tính toán delta (thay đổi chi phí) khi hoán đổi cặp (i, j)

        # Nếu cặp (v_1, v_2) chưa bị cấm trong danh sách Tabu
        if (v_1, v_2) not in tb_list:
            if delta < best_delta_0:  # Nếu delta tốt hơn delta tốt nhất chưa bị cấm, cập nhật
                best_delta_0 = delta
                best_i_0 = i
                best_j_0 = j
        else:  # Nếu cặp (v_1, v_2) bị cấm
            if delta < best_delta_1:  # Nếu delta của cặp bị cấm tốt hơn delta tốt nhất bị cấm
                best_delta_1 = delta
                best_i_1 = i
                best_j_1 = j

    # Kiểm tra nếu move bị cấm lại tốt hơn và có thể cải thiện chi phí thì "phá luật Tabu"
    if best_delta_1 < best_delta_0 and cost + best_delta_1 < best_cost:
        v_1, v_2 = (sol[best_i_1], sol[best_j_1]) if sol[best_i_1] < sol[best_j_1] else (sol[best_j_1], sol[best_i_1])
        tb_list.remove((v_1, v_2))  # Loại bỏ cặp (v_1, v_2) từ đầu danh sách Tabu
        tb_list.append((v_1, v_2))  # Thêm cặp vào cuối danh sách Tabu (giữ cấu trúc FIFO)
        fq_dict[(v_1, v_2)] = fq_dict.get((v_1, v_2), 0) + 1  # Cập nhật tần suất xuất hiện của cặp (v_1, v_2)
        new_sol = get_new_sol(sol, best_i_1, best_j_1)  # Tạo lời giải mới từ việc hoán đổi cặp (best_i_1, best_j_1)
        new_cost = cost + best_delta_1  # Tính chi phí mới
    else:  # Nếu không phá luật Tabu
        if tb_size > 0:
            v_1, v_2 = (sol[best_i_0], sol[best_j_0]) if sol[best_i_0] < sol[best_j_0] else (sol[best_j_0], sol[best_i_0])
            if len(tb_list) == tb_size:  # Nếu danh sách Tabu đầy, xóa phần tử đầu
                tb_list.popleft()
            tb_list.append((v_1, v_2))  # Thêm cặp (v_1, v_2) vào cuối danh sách Tabu
            fq_dict[(v_1, v_2)] = fq_dict.get((v_1, v_2), 0) + 1  # Cập nhật tần suất xuất hiện của cặp (v_1, v_2)
        new_sol = get_new_sol(sol, best_i_0, best_j_0)  # Tạo lời giải mới từ việc hoán đổi cặp tốt nhất chưa bị cấm
        new_cost = cost + best_delta_0  # Tính chi phí mới

    # Trả về lời giải mới, chi phí mới, danh sách Tabu và từ điển tần suất
    return new_sol, new_cost, tb_list, fq_dict
# Từ điển tuần suất chưa xài đâu, nào update thì mới xài

def nearest_neighbor_solution(n, adj_mat, start=None):
    """
    Khởi tạo lời giải bằng thuật toán Nearest Neighbor.
    :param n: số đỉnh
    :param adj_mat: ma trận chi phí
    :param start: đỉnh bắt đầu (mặc định: ngẫu nhiên)
    :return: danh sách hoán vị các đỉnh là tour ban đầu
    """
    if start is None:
        start = random.randint(0, n - 1)

    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        next_city = min(unvisited, key=lambda city: adj_mat[current][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour

def tb(n, adj_mat, tb_size, max_tnm, mut_md, term_count):
    """
    :param n: số lượng đỉnh
    :param adj_mat: ma trận chi phí (khoảng cách giữa các thành phố)
    :param tb_size: kích thước tối đa của danh sách Tabu
    :param max_tnm: số lượng ứng viên được chọn trong lựa chọn giải đấu
    :param mut_md: [get_sol, get delta], phương pháp đột biến, ví dụ như hoán đổi, 2-opt
    :param term_count: số vòng lặp tối đa để dừng thuật toán
    """
    # Khởi tạo
    # sol = list(range(n))
    # random.shuffle(sol)  # Xáo trộn dãy thành phố
    sol = nearest_neighbor_solution(n, adj_mat)
    tb_list = deque([])  # Danh sách Tabu
    fq_dict = {}  # Từ điển tần suất
    best_sol = sol.copy()  # Lời giải tốt nhất
    best_cost = get_cost(n, adj_mat, sol)  # Chi phí tốt nhất

    data = {'cost': deque([]), 'best_cost': deque([])}  # Lưu chi phí qua các vòng lặp
    count = 0

    # Lặp cho đến khi dừng
    while True:
        sol, cost, tb_list, fq_dict = tnm_selection(n, adj_mat, sol,
                                                    max_tnm, mut_md, tb_size,
                                                    tb_list, fq_dict, best_cost)
        # Kiểm tra nếu chi phí giảm và cập nhật lời giải tốt nhất
        if cost < best_cost:
            best_sol = sol
            best_cost = cost
            count = 0
        else:
            count += 1
        data['cost'].append(cost)
        data['best_cost'].append(best_cost)

        # Dừng nếu số vòng không cải thiện vượt quá `term_count`
        if count > term_count:
            break

    # Trả về lời giải tốt nhất, chi phí và dữ liệu về quá trình
    data['fq_dict'] = fq_dict
    return best_sol, best_cost, data

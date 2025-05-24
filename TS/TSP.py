import random
def get_cost(n, adj_mat, sol):
    """
    dùng để tính tổng chi phí của một vòng đi qua tất cả các thành phố một lần trong bài toán TSP
    :param n: Số lượng thành phố, e.g. 2
    :param adj_mat: Ma trận chi phí giữa các thành phố, e.g. [[0,1], [1,0]]
    :param sol: Lời giải, là một hoán vị của các chỉ số thành phố, e.g. [1,0]
    """
    return sum([adj_mat[sol[_]][sol[(_ + 1) % n]] for _ in range(n)])

def get_delta_swap(n, adj_mat, sol, i, j):
    # bef: [..., i-1, i, i+1, ..., j-1, j, j+1] / [...,i-1, i, j, j+1, ...]
    # aft: [..., i-1, j, i+1, ..., j-1, i, j+1] / [...,i-1, j, i, j+1, ...]
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[i - 1]][sol[j]] + adj_mat[sol[j]][sol[(i + 1) % n]] + \
            adj_mat[sol[j - 1]][sol[i]] + adj_mat[sol[i]][sol[(j + 1) % n]] - \
            adj_mat[sol[i - 1]][sol[i]] - adj_mat[sol[i]][sol[(i + 1) % n]] - \
            adj_mat[sol[j - 1]][sol[j]] - adj_mat[sol[j]][sol[(j + 1) % n]]
    if j - i == 1 or i == 0 and j == n - 1:
        delta += 2 * adj_mat[sol[i]][sol[j]]  # symmetrical TSP
    return delta


def get_new_sol_swap(sol, i, j):
    new_sol = sol.copy()
    new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol


def get_delta_2opt(n, adj_mat, sol, i, j):
    # bef: [..., i-1, i, i+1, ..., j-1, j, j+1] / [...,i-1, i, j, j+1, ...] / [i, i+1, ..., j-1, j]
    # aft: [..., i-1, j, j-1, ..., i+1, i, j+1] / [...,i-1, j, i, j+1, ...] / [j, i+1, ..., j-1, i]
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[i - 1]][sol[j]] + adj_mat[sol[i]][sol[(j + 1) % n]] - \
            adj_mat[sol[i - 1]][sol[i]] - adj_mat[sol[j]][sol[(j + 1) % n]]
    if i == 0 and j == n - 1:  # the first two value == 0, while others < 0
        delta = 0
    return delta

def get_new_sol_2opt(sol, i, j):
    new_sol = sol.copy()
    new_sol[i:j+1] = new_sol[i:j+1][::-1]  # notice index + 1 !
    return new_sol

def get_new_sol_vertex_ins(sol, i, j):
    """
    Thực hiện thao tác chèn: lấy đỉnh ở vị trí i, chèn vào trước vị trí j
    Chú ý: j có thể thay đổi nếu i < j do dịch mảng khi pop
    """
    if i == j or i == j - 1:  # không thay đổi
        return sol.copy()
    new_sol = sol.copy()
    city = new_sol.pop(i)
    if i < j:
        j -= 1  # sau khi pop thì vị trí chèn phải lùi lại 1
    new_sol.insert(j, city)
    return new_sol

def get_delta_vertex_ins(n, adj_mat, sol, i, j):
    """
    Tính delta chi phí khi chèn đỉnh i vào trước vị trí j
    """
    if i == j or i == j - 1:
        return 0

    a = sol[i - 1] if i > 0 else sol[-1]
    b = sol[i + 1] if i + 1 < n else sol[0]
    u = sol[j - 1] if j > 0 else sol[-1]
    v = sol[j]

    city = sol[i]

    delta_before = adj_mat[a][city] + adj_mat[city][b] + adj_mat[u][v]
    delta_after = adj_mat[a][b] + adj_mat[u][city] + adj_mat[city][v]

    return delta_after - delta_before


def get_new_sol_or_opt(sol, i, j):
    """
    Cắt đoạn [i:j] và chèn vào một vị trí mới (ngẫu nhiên) không nằm trong đoạn đó.
    i < j, j - i + 1 <= 3
    """
    n = len(sol)
    if j - i + 1 > 3:
        return sol.copy()  # chỉ cho phép đoạn dài <= 3

    segment = sol[i:j+1]
    remain = sol[:i] + sol[j+1:]

    # Chọn vị trí chèn mới
    insert_pos = random.randint(0, len(remain))
    new_sol = remain[:insert_pos] + segment + remain[insert_pos:]

    return new_sol

def get_delta_or_opt(n, adj_mat, sol, i, j):
    """
    Tính delta chi phí nếu cắt đoạn [i:j] (tối đa 3 đỉnh) và chèn vào vị trí khác.
    Do chỉ có i, j nên ta cũng chọn pos ngẫu nhiên như trong get_new_sol_or_opt.
    """
    if j - i + 1 > 3:
        return 0

    segment = sol[i:j+1]
    remain = sol[:i] + sol[j+1:]

    insert_pos = random.randint(0, len(remain))
    # Lấy các đỉnh ảnh hưởng tới chi phí
    a = sol[i - 1] if i > 0 else sol[-1]
    b = sol[j + 1] if j + 1 < n else sol[0]

    u = remain[insert_pos - 1] if insert_pos > 0 else remain[-1]
    v = remain[insert_pos % len(remain)]

    # Trước khi di chuyển:
    cost_before = adj_mat[a][sol[i]] + adj_mat[sol[j]][b] + adj_mat[u][v]
    # Sau khi di chuyển:
    cost_after = adj_mat[a][b] + adj_mat[u][sol[i]] + adj_mat[sol[j]][v]

    return cost_after - cost_before

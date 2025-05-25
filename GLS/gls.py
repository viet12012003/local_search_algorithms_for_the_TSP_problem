# Required libraries
import random
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(file_path):
    with open (file_path, 'r') as f:
        lines = f.readlines()
    idx = 0
    num_datasets = int(lines[idx].strip())
    idx += 1
    datasets = []
    for _ in range(num_datasets):
        num_cities = int(lines[idx].strip())
        idx += 1
        coords = []
        for _ in range(num_cities):
            x, y = map(float, lines[idx].strip().split(','))
            coords.append((x, y))
            idx += 1
        datasets.append(np.array(coords))
    return datasets

def distance_calc(Xdata, city_tour):
    """
    Tính tổng khoảng cách của một hành trình (tour) dựa trên ma trận khoảng cách.
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách giữa các thành phố (n x n)
        Xdata[i][j] là khoảng cách từ thành phố i+1 đến thành phố j+1 (đánh số từ 1)
        
    city_tour : list
        Danh sách chứa 2 phần tử:
        - city_tour[0]: Danh sách các thành phố theo thứ tự thăm
        - city_tour[1]: Chi phí hiện tại (không sử dụng trong hàm này)
        
    Trả về:
    --------
    float
        Tổng khoảng cách của hành trình
    """
    distance = 0
    # Duyệt qua từng cặp thành phố liên tiếp trong hành trình
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        # Cộng dồn khoảng cách giữa thành phố thứ k và k+1, trừ 1 vì Python đánh index từ 0
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]
    return distance

def nearest_neighbor_seed(Xdata):
    """
    Tạo hành trình ban đầu bằng thuật toán láng giềng gần nhất (Nearest Neighbor).
    Thuật toán này chọn thành phố chưa thăm gần nhất làm điểm đến tiếp theo.
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách giữa các thành phố (n x n)
        Xdata[i][j] là khoảng cách từ thành phố i+1 đến thành phố j+1 (đánh số từ 1)
        
    Trả về:
    --------
    list
        Danh sách chứa 2 phần tử:
        - tour: Danh sách các thành phố theo thứ tự thăm (kết thúc bằng thành phố xuất phát)
        - cost: Tổng chi phí (khoảng cách) của hành trình
        Định dạng [tour, cost] này được sử dụng xuyên suốt thuật toán GLS để 
        đảm bảo tính nhất quán và tối ưu hiệu năng (tránh tính toán lại cost nhiều lần)
    """
    n = Xdata.shape[0]  # Số lượng thành phố
    # Danh sách các thành phố chưa thăm (đánh số từ 1 đến n)
    unvisited = list(range(1, n+1))  
    # Bắt đầu từ thành phố 1, đồng thời loại khỏi danh sách chưa thăm
    tour = [unvisited.pop(0)]  

    # Lặp cho đến khi thăm hết các thành phố
    while unvisited:
        last = tour[-1]  # Thành phố hiện tại
        # Tìm thành phố chưa thăm gần nhất
        # key=lambda city: Xdata[last-1, city-1]: sắp xếp theo khoảng cách đến thành phố hiện tại
        next_city = min(unvisited, key=lambda city: Xdata[last-1, city-1])
        tour.append(next_city)  # Thêm vào hành trình
        unvisited.remove(next_city)  # Đánh dấu đã thăm

    # Thêm thành phố xuất phát vào cuối để tạo thành chu trình kín
    tour.append(tour[0])  
    # Tính tổng khoảng cách của hành trình
    cost = distance_calc(Xdata, [tour, 0])
    return [tour, cost]


def build_distance_matrix(coordinates):
    """
    Xây dựng ma trận khoảng cách giữa tất cả các cặp thành phố sử dụng khoảng cách Euclid.
    
    Tham số:
    -----------
    coordinates : numpy.ndarray
        Ma trận 2D chứa tọa độ (x,y) của các thành phố
        Kích thước: (n, 2) với n là số thành phố
        
    Trả về:
    --------
    numpy.ndarray
        Ma trận vuông đối xứng chứa khoảng cách giữa tất cả các cặp thành phố
        Kích thước: (n, n) với n là số thành phố
        dist_matrix[i][j] là khoảng cách từ thành phố i đến thành phố j
    """
    # Lấy số lượng thành phố
    n = coordinates.shape[0]
    
    # Khởi tạo ma trận khoảng cách với giá trị 0
    dist_matrix = np.zeros((n, n))
    
    # Lặp qua tất cả các cặp thành phố
    for i in range(n):
        for j in range(n):
            # Tính khoảng cách Euclid giữa thành phố i và j
            # np.linalg.norm tính căn bậc 2 của tổng bình phương hiệu tọa độ
            dist_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    return dist_matrix


def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    
    # Vẽ đường đi
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    # Đánh dấu thành phố bắt đầu/kết thúc
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    # Đánh dấu thành phố thứ 2
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return


def stochastic_2_opt(Xdata, city_tour):
    """
    Áp dụng phép đảo ngẫu nhiên 2-opt lên một hành trình
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách
    city_tour : list
        [tour, cost] - hành trình hiện tại
        
    Trả về:
    --------
    list
        [new_tour, new_cost] - hành trình mới sau khi đảo
    """
    best_route = copy.deepcopy(city_tour)
    # Chọn ngẫu nhiên 2 điểm cắt không trùng nhau
    i, j = random.sample(range(0, len(city_tour[0])-1), 2)
    if i > j:
        i, j = j, i
    # Đảo ngược đoạn từ i đến j
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
    # Cập nhật thành phố cuối cùng = thành phố đầu
    best_route[0][-1] = best_route[0][0]
    # Tính lại chi phí
    best_route[1] = distance_calc(Xdata, best_route)
    return best_route


def augumented_cost(Xdata, city_tour, penalty, limit):
    """
    Tính chi phí mở rộng (bao gồm cả phạt)
    
    Công thức: cost = chi_phí_thực + λ * tổng_phạt
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách
    city_tour : list
        [tour, cost] - hành trình cần tính
    penalty : numpy.ndarray
        Ma trận phạt hiện tại
    limit : float
        Hệ số λ điều chỉnh ảnh hưởng của phạt
        
    Trả về:
    --------
    float
        Chi phí mở rộng
    """
    augmented = 0
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]
        if c2 < c1:
            c1, c2 = c2, c1
        augmented = augmented + Xdata[c1-1, c2-1] + (limit * penalty[c1-1][c2-1])
    return augmented


def local_search(Xdata, city_tour, penalty, max_attempts=100, limit=1):
    """
    Thực hiện tìm kiếm cục bộ bằng cách sử dụng phép đảo 2-opt ngẫu nhiên.
    Hàm này tìm kiếm các lân cận tốt hơn dựa trên hàm mục tiêu mở rộng.
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách giữa các thành phố (n x n)
    city_tour : list
        [tour, cost] - hành trình hiện tại cần cải thiện
    penalty : numpy.ndarray
        Ma trận phạt hiện tại (n x n)
    max_attempts : int, optional
        Số lần thử tối đa không cải thiện trước khi dừng (mặc định 100)
    limit : float, optional
        Hệ số λ điều chỉnh ảnh hưởng của phạt (mặc định 1)
        
    Trả về:
    --------
    list
        [best_tour, best_cost] - hành trình tốt nhất tìm được
    """
    count = 0  # Đếm số lần thử không cải thiện
    # Tính chi phí mở rộng ban đầu (bao gồm cả phạt)
    ag_cost = augumented_cost(Xdata, city_tour=city_tour, penalty=penalty, limit=limit)
    # Sao chép hành trình hiện tại để tránh thay đổi trực tiếp
    solution = copy.deepcopy(city_tour)
    
    # Lặp cho đến khi đạt max_attempts lần thử không cải thiện
    while count < max_attempts:
        improved = False  # Cờ đánh dấu có cải thiện trong vòng lặp này không
        
        # Thử tìm kiếm 5 lần trong mỗi vòng lặp
        # (Tăng khả năng tìm được lời giải tốt hơn)
        for _ in range(5):
            # 1. Tạo nghiệm mới bằng cách đảo ngẫu nhiên một đoạn hành trình
            candidate = stochastic_2_opt(Xdata, city_tour=solution)
            
            # 2. Tính chi phí mở rộng của nghiệm mới
            candidate_augmented = augumented_cost(
                Xdata, 
                city_tour=candidate, 
                penalty=penalty, 
                limit=limit
            )
            
            # 3. Nếu nghiệm mới tốt hơn (có chi phí thấp hơn)
            if candidate_augmented < ag_cost:
                # Cập nhật nghiệm tốt nhất
                solution = copy.deepcopy(candidate)
                # Cập nhật chi phí tốt nhất
                ag_cost = candidate_augmented
                # Reset bộ đếm vì vừa tìm được nghiệm tốt hơn
                count = 0
                # Đánh dấu đã cải thiện
                improved = True
                # Thoát vòng lặp for để bắt đầu lại với nghiệm mới
                break  
        
        # 4. Nếu không cải thiện sau 5 lần thử
        if not improved:
            count += 1  # Tăng bộ đếm không cải thiện
    
    # Trả về nghiệm tốt nhất tìm được
    return solution


def utility (Xdata, city_tour, penalty, limit = 1):
    """
    Tính utility cho mỗi cạnh trong hành trình
    
    Utility = chi_phí_thực / (1 + số_lần_phạt_cạnh_đang_xét)
    Cạnh có utility cao là ứng cử viên để bị phạt
    
    Trả về:
    --------
    list
        Danh sách utility của từng cạnh
    """
    utilities = [0 for i in city_tour[0]]
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]
        if c2 < c1:
            c1, c2 = c2, c1
        utilities[i] = Xdata[c1-1, c2-1] /(1 + penalty[c1-1][c2-1])
    return utilities


def update_penalty(penalty, city_tour, utilities):
    """
    Cập nhật ma trận phạt dựa trên utility
    
    Tăng phạt cho cạnh có utility cao nhất
    """
    max_utility = max(utilities)
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]
        if c2 < c1:
            c1, c2 = c2, c1
        if (utilities[i] == max_utility):
            penalty[c1-1][c2-1] = penalty[c1-1][c2-1] + 1
    return penalty

# Function: Guided Search
def guided_search(Xdata, city_tour, alpha=0.1, local_search_optima=1000, max_attempts=50):
    """
    Thực hiện GLS (GLS)
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách
    city_tour : list
        [tour, cost] - hành trình ban đầu
    alpha : float
        Hệ số điều chỉnh cường độ phạt
    local_search_optima : int
        Số lần lặp tối đa
    max_attempts : int
        Số lần thử tối đa không cải thiện trong local search
        
    Trả về:
    --------
    tuple
        (best_solution, best_cost) - kết quả tốt nhất tìm được
    """
    n = Xdata.shape[0]
    penalty = np.zeros((n, n))  # Khởi tạo ma trận phạt
    limit = alpha * np.mean(Xdata)  # Tính λ = α * chi_phí_trung_bình
    best_solution = copy.deepcopy(city_tour)
    best_cost = distance_calc(Xdata, best_solution)
    no_improvement_count = 0
    max_no_improvement = 20  # Số vòng không cải thiện tối đa trước khi dừng
    
    for i in range(local_search_optima):
        # 1. Tìm kiếm cục bộ với hàm mục tiêu mở rộng
        solution = local_search(Xdata, city_tour=best_solution, 
                              penalty=penalty, 
                              max_attempts=max_attempts, 
                              limit=limit)
        
        # 2. Tính utility cho các cạnh
        utilities = utility(Xdata, city_tour=solution, 
                          penalty=penalty, limit=limit)
        
        # 3. Cập nhật ma trận phạt
        penalty = update_penalty(penalty, solution, utilities)
        
        # 4. Đánh giá chi phí thực
        cost = distance_calc(Xdata, solution)
        
        # 5. Cập nhật nghiệm tốt nhất
        if cost < best_cost:
            best_cost = cost
            best_solution = copy.deepcopy(solution)
            no_improvement_count = 0
            print(f"  GLS loop {i+1}/{local_search_optima}, new best cost={best_cost}")
        else:
            no_improvement_count += 1
        
        # 6. Kiểm tra điều kiện dừng sớm
        if no_improvement_count >= max_no_improvement and i >= max_attempts:
            print(f"  No improvement for {max_no_improvement} loops, stopping early")
            print("-----------------------------------------------------------------")
            break
    return best_solution, best_cost


def run_multiple_gls(Xdata, iterations=30, alpha=0.1, seed_func=nearest_neighbor_seed, max_attempts=50):
    """
    Thực hiện nhiều lần GLS để tìm nghiệm tốt nhất
    
    Tham số:
    -----------
    Xdata : numpy.ndarray
        Ma trận khoảng cách
    iterations : int
        Số lần lặp
    alpha : float
        Hệ số điều chỉnh cường độ phạt
    seed_func : function
        Hàm tạo tour khởi tạo
    max_attempts : int
        Số lần thử tối đa không cải thiện trong local search
        
    Trả về:
    --------
    tuple
        (best_solution, best_cost) - kết quả tốt nhất tìm được
    """
    # Tìm tour khởi tạo tốt nhất
    best_seed = seed_func(Xdata)
    
    costs = []
    times = []
    best_cost = float('inf')
    best_tour = None
    
    for it in range(iterations):
        seed = copy.deepcopy(best_seed)
        start = time.time()
        solution, cost = guided_search(
            Xdata, 
            city_tour=seed, 
            alpha=alpha, 
            local_search_optima=1000, 
            max_attempts=max_attempts
        )
        end = time.time()
        costs.append(cost)
        times.append(end - start)
        if cost < best_cost:
            best_cost = cost
            best_tour = solution
        print(f"  Iteration {it+1}/{iterations} done, cost={cost}, time={end-start:.2f}s")
    avg_cost = np.mean(costs)
    avg_time = np.mean(times)
    return best_cost, avg_cost, avg_time, best_tour


# ================================= MAIN ===================================
def main():
    file_path = "D:\ADA\local_search_algorithms_for_the_TSP_problem\data.txt"
    bks_values = [426, 538, 21282, 6528, 29368, 49135, 42029, 15281, 107217, 6773]
    alpha = 0.1
    results = []
    
    print("Đang tải dữ liệu...")
    datasets = preprocess_data(file_path)
    
    for idx, coords in enumerate(datasets):
        print(f"\n=== Xử lý Dataset {idx + 1} ===")
        Xdata = build_distance_matrix(coords)
        print(f"  Đang chạy GLS cho dataset {idx + 1}...")
        best_cost, avg_cost, avg_time, best_tour = run_multiple_gls(
            Xdata, 
            iterations=30,
            alpha=alpha, 
            seed_func=nearest_neighbor_seed,
            max_attempts=100
        )
        print(f"  Best cost: {best_cost}")
        print(f"  Average cost: {avg_cost}")
        print(f"  Average time: {avg_time:.4f} s")
        plt.figure(figsize=(10, 8))
        plot_tour_coordinates(coords, best_tour)
        plt.title(f"Dataset {idx+1} - Best Tour (Cost: {best_cost:.2f})")
        plt.savefig(f"dataset_{idx+1}_tour.png")
        plt.close()
       
        results.append({
            'dataset': idx + 1,
            'alpha': alpha,
            'best_cost': round(best_cost, 2),
            'avg_cost': round(avg_cost, 2),
            'avg_time': round(avg_time, 2),
            'BKS': bks_values[idx]
        })
    
    for result in results:
        bks = result['BKS']
        best_cost = result['best_cost']
        avg_cost = result['avg_cost']
        result['Error (%)'] = round((avg_cost - bks) / bks * 100, 2)
        result['PE (%)'] = round((avg_cost - best_cost) / best_cost * 100, 2)
    
    df = pd.DataFrame(results)
    df = df[['dataset', 'alpha', 'best_cost', 'avg_cost', 'avg_time', 
             'BKS', 'Error (%)', 'PE (%)']]
    
    print("\n=== Kết quả thực nghiệm GLS ===")
    for dataset in sorted(df['dataset'].unique()):
        print(f"\n=== Dataset {dataset} ===")
        print(df[df['dataset'] == dataset].to_string(index=False))
   
    output_file = "GLS_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nĐã lưu kết quả vào file: {output_file}")

if __name__ == "__main__":
    main()


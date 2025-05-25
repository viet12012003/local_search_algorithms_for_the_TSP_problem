import math
import os
import random
import time

from typing import List, Callable, Tuple

from LBSA.model import *
from LBSA.config import *

inverse_count = 0
insert_count = 0
swap_count = 0


def evaluate_distance(a: City, b: City) -> float:
    return math.sqrt(abs(a.x - b.x) ** 2 + abs(a.y - b.y) ** 2)


def evaluate_solution(cities: List[City], solution: Solution) -> float:
    solution_pair = zip(solution, solution[1:] + solution[:1])
    distances = [evaluate_distance(cities[a], cities[b]) for a, b in solution_pair]
    total = sum(distances)
    return total


def read_data(file_location: str) -> List[TestCase]:
    if os.path.isfile(file_location):
        with open(file_location) as file:  # Use file to refer to the file object
            test_case_count = int(file.readline())
            test_case = list()
            for _ in range(test_case_count):
                city_count = int(file.readline())
                cities = list()
                for count in range(city_count):
                    line = file.readline().split(',')
                    x, y = list(map(lambda x: float(x), line))
                    cities.append(City(str(count), x, y))
                test_case.append(TestCase(cities))

        return test_case
    return list()


def generate_2_random_index(cities: List[City]) -> int:
    return random.sample(range(len(cities)), 2)


def generate_random_probability_r() -> float:
    return random.uniform(0.00000001, 1)


"""
given 
index [i, j] = [3, 6]
a list    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
will return [0, 1, 5, 4, 3, 2, 6, 7, 8, 9]
"""


def inverse_solution(old_solution: Solution, i: int, j: int) -> Solution:
    numbers = [i, j]
    numbers.sort()
    i, j = numbers
    return old_solution[:i - 1] + old_solution[i - 1:j][::-1] + old_solution[j:]


"""
given 
index [i, j] = [3, 6]
a list    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
will return [0, 1, 5, 2, 3, 4, 6, 7, 8, 9]
"""


def insert_solution(old_solution: Solution, i: int, j: int) -> Solution:
    new_solution = old_solution[:]
    element = new_solution.pop(j - 1)
    new_solution.insert(i - 1, element)
    return new_solution


"""
given 
index [i, j] = [3, 6]
a list    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
will return [0, 1, 5, 3, 4, 2, 6, 7, 8, 9]
"""


def swap_solution(old_solution: Solution, i: int, j: int) -> Solution:
    new_solution = old_solution[:]
    temp = new_solution[i - 1]
    new_solution[i - 1] = new_solution[j - 1]
    new_solution[j - 1] = temp
    return new_solution


def create_new_solution(cities: List[City], old_solution: Solution, i_test: int = -1, j_test: int = -1) -> Solution:
    global insert_count, inverse_count, swap_count

    # helper for unit test, so number is not random
    i, j = i_test, j_test

    if i == -1 or j == -1:
        i, j = generate_2_random_index(cities)

    inverse_opt = inverse_solution(old_solution, i, j)
    insert_opt = insert_solution(old_solution, i, j)
    swap_opt = swap_solution(old_solution, i, j)

    evaluation = [evaluate_solution(cities, inverse_opt), evaluate_solution(cities, insert_opt),
                  evaluate_solution(cities, swap_opt)]
    index = evaluation.index(min(evaluation))

    if index == 0:
        inverse_count += 1
        return inverse_opt
    elif index == 1:
        insert_count += 1
        return insert_opt
    else:
        swap_count += 1
        return swap_opt


def calculate_bad_result_acceptance_probability(tmax: float, evaluation_new_solution: float,
                                                evaluation_old_solution: float) -> float:
    return math.exp(-(evaluation_new_solution - evaluation_old_solution) / tmax)


def calculate_new_temperature(r_probability: float, old_temperature: float, evaluation_new_solution: float,
                              evaluation_old_solution: float) -> float:
    return (old_temperature - (evaluation_new_solution - evaluation_old_solution)) / math.log(r_probability)


"""
create initial temperature
temperature_list_length is how many try we find the initial temp
initial_acc_probability should be 0..1
"""


def create_initial_temp(cities: List[City], temperature_list_length: int, initial_acc_probability: float) -> List[
    float]:
    solution = list(range(len(cities)))
    temperature_list = [2]

    for _ in range(temperature_list_length):
        old_solution = solution
        new_solution = create_new_solution(cities, solution)

        new_evaluation = evaluate_solution(cities, new_solution)
        old_evaluation = evaluate_solution(cities, old_solution)

        if new_evaluation < old_evaluation:
            solution = new_solution

        t = (- abs(new_evaluation - old_evaluation)) / math.log(initial_acc_probability)
        temperature_list.append(t)

    return temperature_list


"""
result should be look like this: [0, 1, 7, 9, 5, 4, 8, 6, 2, 3]
"""


def run_lbsa(cities: List[City], M: int, K: int, temperature_list_length: int, initial_acc_probability: float) \
        -> Tuple[Solution, List[float]]:
    temperature_list = create_initial_temp(cities, temperature_list_length, initial_acc_probability)
    temperature_list = [max(temperature_list)]
    current_solution = list(range(len(cities)))
    evaluation_result_list = list()  # debugging purpose
    best_solution = current_solution
    best_evaluation = evaluate_solution(cities, best_solution)

    k = 0
    while k <= K:
        # clean temparature list
        k += 1
        t = 0
        m = 0
        c = 0

        while m <= M:
            m += 1
            new_solution = create_new_solution(cities, current_solution)

            new_evaluation = evaluate_solution(cities, new_solution)
            current_evaluation = evaluate_solution(cities, current_solution)

            if new_evaluation < current_evaluation:
                current_solution = new_solution
                current_evaluation = new_evaluation

                if current_evaluation < best_evaluation:
                    best_solution = current_solution
                    best_evaluation = current_evaluation

            else:
                p = calculate_bad_result_acceptance_probability(max(temperature_list), new_evaluation,
                                                                current_evaluation)
                r = generate_random_probability_r()

                if r > p:
                    t = calculate_new_temperature(r, t, new_evaluation, current_evaluation)
                    temperature_list.append(t)
                    current_solution = new_solution
                    c += 1

                    if current_evaluation < best_evaluation:
                        best_solution = current_solution
                        best_evaluation = current_evaluation

            evaluation_result_list.append(current_evaluation)

        if c > 0:
            t = t / c
            temperature_list.remove(max(temperature_list))
            temperature_list.append(t / c)

    return best_solution, evaluation_result_list


if __name__ == '__main__':
    DATA_SET = read_data(DATA_FILE)

    for idx, test in enumerate(DATA_SET, start=1):
        total_distance = 0
        total_time = 0
        num_runs = 30  # Số lần chạy
        best_solution_overall = None  # Lưu lộ trình tốt nhất
        best_distance = float('inf')  # Chi phí tốt nhất (càng nhỏ càng tốt)

        for _ in range(num_runs):
            start = time.time()
            solution, result_list = run_lbsa(test.cities, M, K, T_LENGTH, INIT_P)
            end = time.time()

            duration = end - start
            total_time += duration

            distance = evaluate_solution(test.cities, solution)
            total_distance += distance

            # Cập nhật chi phí tốt nhất và lộ trình tốt nhất
            if distance < best_distance:
                best_distance = distance
                best_solution_overall = solution

        # Tính giá trị trung bình
        average_distance = total_distance / num_runs
        avg_time = total_time / num_runs

        print(f"Test case {idx}:")
        print(f'Average cost over {num_runs} runs: {average_distance}')
        print(f'Best cost over {num_runs} runs: {best_distance}')
        print(f'Best route corresponding to best cost: {best_solution_overall}')
        print(f'Average time over {num_runs} runs: {avg_time:.2f} seconds')

        print(f"Number of times inverse_opt was used: {inverse_count}")
        print(f"Number of times insert_opt was used: {insert_count}")
        print(f"Number of times swap_opt was used: {swap_count}")
        print(f"Total transformations used: {inverse_count + insert_count + swap_count}")

        if SHOW_VISUAL:
            import matplotlib.pyplot as plt

            # Vẽ đồ thị cho lộ trình tốt nhất
            for city in test.cities:
                plt.plot(city.x, city.y, color='r', marker='o')

            # Thêm điểm đầu để khép kín chu trình
            best_solution_overall.append(best_solution_overall[0])
            x_points = [test.cities[i].x for i in best_solution_overall]
            y_points = [test.cities[i].y for i in best_solution_overall]

            plt.plot(x_points, y_points, linestyle='--', color='b')
            plt.title("Best Route")
            plt.show()  # Visualization of the best route

            # Vẽ đồ thị kết quả đánh giá của lần chạy cuối cùng
            plt.plot(list(range(len(result_list))), result_list, linestyle='-', color='b')
            plt.title("Evaluation per Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.grid(True)
            plt.show()

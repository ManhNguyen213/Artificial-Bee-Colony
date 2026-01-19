import random
import numpy as np
import matplotlib.pyplot as plt 

# Hàm mục tiêu: Hàm Rastrigin
def objective_function(x):
    x = np.array(x)
    D = len(x)
    return 10*D + np.sum(x**2 - 10*np.cos(2*np.pi*x))

# Tính Fitness
def calculate_fitness(f):
    return 1/(1 + f) if f >= 0 else 1 + abs(f)

class FoodSource:
    def __init__(self, position, func):
        self.position = np.array(position, dtype=float)
        self.obj = func(self.position)
        self.fitness = calculate_fitness(self.obj)
        self.trial = 0

class ABC:
    def __init__(self, func, bounds, pop_size=20, max_iter=500, limit=50):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit

        # Quần thể
        self.population = [
            FoodSource([random.uniform(b[0], b[1]) for b in bounds], func)
            for _ in range(pop_size)
        ]

        self.best = min(self.population, key=lambda f: f.obj)
        self.history = []

    # Tạo nghiệm mới
    def generate_neighbor(self, i):
        xi = self.population[i].position.copy()
        k = random.choice([a for a in range(self.pop_size) if a != i])
        xk = self.population[k].position
        j = random.randint(0, len(self.bounds) - 1)
        phi = random.uniform(-1, 1)
        v = xi.copy()
        v[j] = xi[j] + phi*(xi[j] - xk[j])
        low, high = self.bounds[j]
        v[j] = np.clip(v[j], low, high)
        return v

    # Xác suất chọn nguồn thức ăn
    def calculate_prob(self):
        fits = np.array([sol.fitness for sol in self.population])
        return (fits / np.sum(fits))

    # Thuật toán ABC
    def run(self):
        for it in range(self.max_iter):

            # Pha ong thợ (Employed Bees)
            for i in range(self.pop_size):
                v = self.generate_neighbor(i)
                new_obj = self.func(v)
                new_fit = calculate_fitness(new_obj)
                if new_fit > self.population[i].fitness:
                    self.population[i].position = v
                    self.population[i].obj = new_obj
                    self.population[i].fitness = new_fit
                    self.population[i].trial = 0
                else:
                    self.population[i].trial += 1

            # Pha ong quan sát (Onlooker Bees)
            prob = self.calculate_prob()
            for _ in range(self.pop_size):
                i = np.random.choice(range(self.pop_size), p=prob)
                v = self.generate_neighbor(i)
                new_obj = self.func(v)
                new_fit = calculate_fitness(new_obj)
                if new_fit > self.population[i].fitness:
                    self.population[i].position = v
                    self.population[i].obj = new_obj
                    self.population[i].fitness = new_fit
                    self.population[i].trial = 0
                else:
                    self.population[i].trial += 1

            # Pha ong trinh sát (Scout Bees)
            worst = max(self.population, key=lambda f: f.trial)
            if worst.trial > self.limit:
                new_pos = np.array([random.uniform(b[0], b[1]) for b in self.bounds])
                worst.position = new_pos
                worst.obj = self.func(new_pos)
                worst.fitness = calculate_fitness(worst.obj)
                worst.trial = 0

            # Cập nhật nghiệm tốt nhất
            current_best = min(self.population, key=lambda f: f.obj)
            if current_best.obj < self.best.obj:
                self.best = current_best

            self.history.append(self.best.obj)

            # In giá trị mỗi vòng
            if it < 50:
                print(f"Iter {it}: best = {self.best.obj:.6f}, x = {self.best.position}")

        return self.best.position, self.best.obj

# ================================
# CHẠY THUẬT TOÁN
# ================================
if __name__ == "__main__":
    bounds = [(-5, 5), (-5, 5)]
    abc = ABC(objective_function, bounds, pop_size=20, max_iter=500, limit=50)
    best_pos, best_val = abc.run()
    print("\nFINAL BEST =", best_pos)
    print("FINAL OBJ  =", best_val)

    # VẼ ĐỒ THỊ HỘI TỤ
    history_100 = abc.history[:50]
    plt.figure(figsize=(10,6))
    plt.plot(history_100, color='blue', marker='o', markersize=3, linewidth=1)
    plt.title("Convergence Curve of ABC (First 100 iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.grid(True)
    plt.show()

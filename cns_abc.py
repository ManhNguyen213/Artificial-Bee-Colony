import random
import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu: Rastrigin
def objective_function(x):
    x = np.array(x)
    D = len(x)
    return 10*D + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def calculate_fitness(f):
    return 1/(1 + f) if f >= 0 else 1 + abs(f)

class FoodSource:
    def __init__(self, position, func):
        self.position = np.array(position, dtype=float)
        self.obj = func(self.position)
        self.fitness = calculate_fitness(self.obj)
        self.trial = 0

# Chaotic Initialization (Bernoulli)
def chaotic_initialization(bounds, pop_size):
    beta = 0.7  # Hệ số chaos
    D = len(bounds)
    population = []
    for _ in range(pop_size):
        x = []
        for j in range(D):
            z = random.random()
            if z <= 1 - beta:
                zB = z / (1 - beta)
            else:
                zB = (z - 1 + beta) / beta
            # Cơ chế loại trừ (mutual exclusion): tạo 2 giá trị, chọn nhỏ hơn
            xi1 = bounds[j][0] + zB * (bounds[j][1] - bounds[j][0])
            xi2 = bounds[j][1] - zB * (bounds[j][1] - bounds[j][0])
            x.append(min(xi1, xi2))
        population.append(np.array(x))
    return population

# Thuật toán CNSABC
class CNSABC:
    def __init__(self, func, bounds, pop_size=20, max_iter=500, limit=50):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit

        # Khởi tạo quần thể với chaotic initialization
        initial_positions = chaotic_initialization(bounds, pop_size)
        self.population = [FoodSource(pos, func) for pos in initial_positions]

        self.best = min(self.population, key=lambda f: f.obj)
        self.history = []

    # Tìm kiếm lân cận
    # Ong thợ: khám phá toàn cục
    # Ong quan sát: khai thác cục bộ
    def generate_neighbor(self, i, mode='employed', iter_num=0):
        xi = self.population[i].position.copy()
        k = random.choice([a for a in range(self.pop_size) if a != i])
        xk = self.population[k].position
        j = random.randint(0, len(self.bounds) - 1)
        phi = random.uniform(-1, 1)
        v = xi.copy()

        # Hệ số nén thích ứng
        cp = 2 / (1 + iter_num)  # giảm dần khi tiến hóa

        # Trọng số ngẫu nhiên
        psi1 = 1.5 * random.random()
        psi2 = 6 * random.random()

        xbest = self.best.position

        if mode == 'employed':  # Ong thợ: khám phá toàn cục
            v[j] = xi[j] + phi*(xi[j] - xk[j]) + psi1*(xbest[j] - xi[j])
        else:  # Onlooker bees: local exploitation
            v[j] = cp*xi[j] + phi*(xi[j] - xk[j]) + psi2*(xbest[j] - xi[j])

        # Giới hạn biên
        low, high = self.bounds[j]
        v[j] = np.clip(v[j], low, high)
        return v

    def calculate_prob(self):
        fits = np.array([sol.fitness for sol in self.population])
        return fits / np.sum(fits)

    def run(self):
        for it in range(self.max_iter):

            # Pha ong thợ (Employed bees)
            for i in range(self.pop_size):
                v = self.generate_neighbor(i, mode='employed', iter_num=it)
                new_obj = self.func(v)
                new_fit = calculate_fitness(new_obj)
                if new_fit > self.population[i].fitness:
                    self.population[i].position = v
                    self.population[i].obj = new_obj
                    self.population[i].fitness = new_fit
                    self.population[i].trial = 0
                else:
                    self.population[i].trial += 1

            # Pha ong quan sát (Onlooker bees)
            prob = self.calculate_prob()
            for _ in range(self.pop_size):
                i = np.random.choice(range(self.pop_size), p=prob)
                v = self.generate_neighbor(i, mode='onlooker', iter_num=it)
                new_obj = self.func(v)
                new_fit = calculate_fitness(new_obj)
                if new_fit > self.population[i].fitness:
                    self.population[i].position = v
                    self.population[i].obj = new_obj
                    self.population[i].fitness = new_fit
                    self.population[i].trial = 0
                else:
                    self.population[i].trial += 1

            # Pha ong trinh sát (Scout bees)
            worst = max(self.population, key=lambda f: f.trial)
            if worst.trial > self.limit:
                new_pos = np.array([random.uniform(b[0], b[1]) for b in self.bounds])
                worst.position = new_pos
                worst.obj = self.func(new_pos)
                worst.fitness = calculate_fitness(worst.obj)
                worst.trial = 0

            # Sustained bees: liên tục khai thác solution tốt nhất
            for sb in range(self.pop_size//5):  # khoảng 20% số ong
                v = self.best.position + (1/(1+it))*np.random.rand(len(self.bounds))*(self.best.position - self.population[sb].position)
                new_obj = self.func(v)
                new_fit = calculate_fitness(new_obj)
                if new_fit > self.population[sb].fitness:
                    self.population[sb].position = v
                    self.population[sb].obj = new_obj
                    self.population[sb].fitness = new_fit
                    self.population[sb].trial = 0

            # Cập nhật nghiệm tốt nhất
            current_best = min(self.population, key=lambda f: f.obj)
            if current_best.obj < self.best.obj:
                self.best = current_best

            self.history.append(self.best.obj)

            if it < 50:
                print(f"Iter {it}: best = {self.best.obj:.6f}, x = {self.best.position}")

        return self.best.position, self.best.obj
    
# ============================
# CHẠY THUẬT TOÁN CNSABC
# ============================
if __name__ == "__main__":
    bounds = [(-5, 5), (-5, 5)]
    cnsabc = CNSABC(objective_function, bounds, pop_size=20, max_iter=500, limit=50)
    best_pos, best_val = cnsabc.run()
    print("\nFINAL BEST =", best_pos)
    print("FINAL OBJ  =", best_val)

    # VẼ ĐỒ THỊ HỘI TỤ
    plt.figure(figsize=(10,6))
    plt.plot(cnsabc.history[:50], color='red', marker='o', markersize=3, linewidth=1)
    plt.title("Convergence Curve of CNSABC (First 50 iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.grid(True)
    plt.show()

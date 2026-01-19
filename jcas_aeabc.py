import random
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Hàm mục tiêu JCAS
# ======================================================
def beamforming_objective(x, A, v, D):
    x = np.asarray(x, dtype=float)

    M = A.shape[1]
    w = x[:M] + 1j * x[M:2*M]

    # Chuẩn hoá: ||w||_2 = 1
    w = w / np.linalg.norm(w)

    Aw  = A @ w
    DAw = D @ Aw
    Dv  = D @ v

    # Hệ số c_s
    numerator   = np.real(np.vdot(DAw, Dv))
    denominator = np.real(np.vdot(DAw, DAw)) + 1e-12
    c_s = numerator / denominator

    # Sai số xấp xỉ bình phương
    error = D @ (c_s * Aw - v)
    return np.real(np.vdot(error, error))


# Hàm Fitness
def calculate_fitness(cost):
    return 1.0 / (cost + 1e-12)


# ======================================================
# Adaptive Exploration Artificial Bee Colony (AEABC)
# ======================================================
class AEABC:
    def __init__(self, obj_func, bounds, pop_size=30, max_iter=200, limit=40):
        self.obj_func = obj_func
        self.bounds   = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit    = limit

        self.dim = len(bounds)

        # Khởi tạo quần thể ban đầu
        self.population = [self.random_solution() for _ in range(pop_size)]
        self.fitness    = np.zeros(pop_size)
        self.trial      = np.zeros(pop_size)

        self.best_sol  = None
        self.best_cost = np.inf
        self.history   = []

        self.evaluate()

    # Khởi tạo nghiệm ngẫu nhiên
    def random_solution(self):
        x = np.array([random.uniform(b[0], b[1]) for b in self.bounds])

        M = len(x) // 2
        w = x[:M] + 1j * x[M:2*M]
        w = w / np.linalg.norm(w)

        return np.concatenate([np.real(w), np.imag(w)])

    # Đánh giá quần thể và cập nhật nghiệm tốt nhất
    def evaluate(self):
        for i in range(self.pop_size):
            cost = self.obj_func(self.population[i])
            self.fitness[i] = calculate_fitness(cost)

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_sol  = self.population[i].copy()

    # Khoảng cách Euclid trong AEABC
    def distance(self, xi, xk):
        return np.linalg.norm(xi - xk)

    # Tạo nghiệm mới theo AEABC
    def generate_neighbor(self, i):
        k = random.choice([j for j in range(self.pop_size) if j != i])

        xi = self.population[i].copy()
        xk = self.population[k]

        j = random.randint(0, self.dim - 1)
        phi = random.uniform(-1, 1)

        v = xi.copy()
        v[j] = xi[j] + phi * (xi[j] - xk[j])

        # Adaptive Exploration
        d = self.distance(xi, xk)
        Pd = np.exp(-1.0 / d) if d > 0 else 0.0
        r = random.random()

        if r > Pd:
            v[j] = r * v[j]

        # Đảm bảo nghiệm nằm trong miền ràng buộc
        low, high = self.bounds[j]
        v[j] = np.clip(v[j], low, high)

        # Chuẩn hóa công suất (JCAS)
        M = len(v) // 2
        w = v[:M] + 1j * v[M:2*M]
        w = w / np.linalg.norm(w)

        return np.concatenate([np.real(w), np.imag(w)])

    # Thuật toán AEABC
    def run(self):
        for it in range(self.max_iter):

            # Pha ong thợ (Employed Bees)
            for i in range(self.pop_size):
                v = self.generate_neighbor(i)
                cost_v = self.obj_func(v)
                fit_v  = calculate_fitness(cost_v)

                if fit_v > self.fitness[i]:
                    self.population[i] = v
                    self.fitness[i]    = fit_v
                    self.trial[i]      = 0
                else:
                    self.trial[i] += 1

            # Pha ong quan sát (Onlooker Bees)
            prob = self.fitness / np.sum(self.fitness)
            for _ in range(self.pop_size):
                i = np.random.choice(range(self.pop_size), p=prob)
                v = self.generate_neighbor(i)
                cost_v = self.obj_func(v)
                fit_v  = calculate_fitness(cost_v)

                if fit_v > self.fitness[i]:
                    self.population[i] = v
                    self.fitness[i]    = fit_v
                    self.trial[i]      = 0
                else:
                    self.trial[i] += 1

            # Pha ong trinh sát (Scout Bees)
            worst = np.argmax(self.trial)
            if self.trial[worst] > self.limit:
                self.population[worst] = self.random_solution()
                self.trial[worst] = 0

            self.evaluate()
            self.history.append(self.best_cost)

            print(f"Iter {it:3d} | Best LS error = {self.best_cost:.4e}")

        return self.best_sol, self.best_cost


# ======================================================
# Thiết lập bài toán JCAS Multibeam
# ======================================================
M = 4
K = 50

theta = np.linspace(-np.pi/2, np.pi/2, K)

A = np.exp(1j * np.outer(np.sin(theta), np.arange(M)))
v = np.exp(-(theta / 0.2)**2)

D = np.eye(K)
for k in range(K):
    D[k, k] = 5.0 if abs(theta[k]) < 0.15 else 1.0

bounds = [(-5, 5)] * (2 * M)

aeabc = AEABC(
    obj_func=lambda x: beamforming_objective(x, A, v, D),
    bounds=bounds,
    pop_size=30,
    max_iter=200,
    limit=40
)

best_x, best_cost = aeabc.run()
print("\nFINAL BEST LS ERROR =", best_cost)


# VẼ ĐỒ THỊ HỘI TỤ
plt.figure(figsize=(10,6))
plt.plot(aeabc.history[:100], color='green', marker='o', markersize=3, linewidth=1, label='AEABC')
plt.xlabel("Iteration")
plt.ylabel("Best LS Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

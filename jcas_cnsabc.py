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

    w = w / np.linalg.norm(w)

    Aw  = A @ w
    DAw = D @ Aw
    Dv  = D @ v

    numerator   = np.real(np.vdot(DAw, Dv))
    denominator = np.real(np.vdot(DAw, DAw)) + 1e-12
    c_s = numerator / denominator

    error = D @ (c_s * Aw - v)
    return np.real(np.vdot(error, error))


# Hàm Fitness
def calculate_fitness(cost):
    return 1.0 / (cost + 1e-12)


# ======================================================
# Chaotic Initialization (Bernoulli Map)
# ======================================================
def chaotic_initialization(bounds, pop_size):
    beta = 0.7
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

            xi1 = bounds[j][0] + zB * (bounds[j][1] - bounds[j][0])
            xi2 = bounds[j][1] - zB * (bounds[j][1] - bounds[j][0])
            x.append(min(xi1, xi2))

        population.append(np.array(x, dtype=float))

    return population


# ======================================================
# Chaos-based Neighborhood Search ABC (CNSABC)
# ======================================================
class CNSABC:
    def __init__(self, obj_func, bounds, pop_size=30, max_iter=200, limit=40):
        self.obj_func = obj_func
        self.bounds   = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit    = limit

        self.dim = len(bounds)

        self.population = chaotic_initialization(bounds, pop_size)
        self.fitness = np.zeros(pop_size)
        self.trial   = np.zeros(pop_size)

        self.best_sol  = None
        self.best_cost = np.inf
        self.history   = []

        self.evaluate()

    # Chuẩn hóa công suất nghiệm beamforming
    def normalize_solution(self, x):
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

    # Tạo nghiệm mới
    def generate_neighbor(self, i, mode='employed', it=0):
        xi = self.population[i]
        k = random.choice([j for j in range(self.pop_size) if j != i])
        xk = self.population[k]

        j = random.randint(0, self.dim - 1)
        phi = random.uniform(-1, 1)

        cp   = 2 / (1 + it)
        psi1 = 1.5 * random.random()
        psi2 = 6.0 * random.random()

        v = xi.copy()

        if mode == 'employed':  # global exploration
            v[j] = xi[j] + phi*(xi[j] - xk[j]) + psi1*(self.best_sol[j] - xi[j])
        else:  # local exploitation
            v[j] = cp*xi[j] + phi*(xi[j] - xk[j]) + psi2*(self.best_sol[j] - xi[j])

        low, high = self.bounds[j]
        v[j] = np.clip(v[j], low, high)

        return self.normalize_solution(v)

    # Thuật toán CNSABC
    def run(self):
        for it in range(self.max_iter):

            # Pha ong thợ (Employed Bees)
            for i in range(self.pop_size):
                v = self.generate_neighbor(i, 'employed', it)
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
                v = self.generate_neighbor(i, 'onlooker', it)
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
                self.population[worst] = self.normalize_solution(
                    np.array([random.uniform(b[0], b[1]) for b in self.bounds])
                )
                self.trial[worst] = 0

            self.evaluate()
            self.history.append(self.best_cost)

            print(f"Iter {it:3d} | Best LS error = {self.best_cost:.4e}")

        return self.best_sol, self.best_cost


# ======================================================
# Thiết lập bài toán JCAS Multibeam
# ======================================================
if __name__ == "__main__":

    M = 4
    K = 50

    theta = np.linspace(-np.pi/2, np.pi/2, K)

    A = np.exp(1j * np.outer(np.sin(theta), np.arange(M)))
    v = np.exp(-(theta / 0.2)**2)

    D = np.eye(K)
    for k in range(K):
        D[k, k] = 5.0 if abs(theta[k]) < 0.15 else 1.0

    bounds = [(-5, 5)] * (2 * M)

    cnsabc = CNSABC(
        obj_func=lambda x: beamforming_objective(x, A, v, D),
        bounds=bounds,
        pop_size=30,
        max_iter=200,
        limit=40
    )

    best_x, best_cost = cnsabc.run()
    print("\nFINAL BEST LS ERROR =", best_cost)

    # VẼ ĐỒ THỊ HỘI TỤ
    plt.figure(figsize=(10,6))
    plt.plot(cnsabc.history[:100], marker='o', markersize=3, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

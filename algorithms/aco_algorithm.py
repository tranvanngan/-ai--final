"""
Ant Colony Optimization (ACO) implementation for Knapsack Problem
"""
import random
from PyQt6.QtCore import QThread, pyqtSignal


class ACOWorker(QThread):
    progress_updated = pyqtSignal(int, float, float)
    finished = pyqtSignal(list, float, float, list)

    def __init__(self, items, capacity, num_ants, max_iter, alpha, beta, evaporation_rate):
        super().__init__()
        self.items = items
        self.capacity = capacity
        self.num_ants = max(1, num_ants)
        self.max_iter = max(1, max_iter)
        self.alpha = max(0.0, alpha)
        self.beta = max(0.0, beta)
        self.rho = min(0.99, max(0.0, evaporation_rate))
        self.num_items = len(items)

        # initialize pheromone
        initial_tau = 1.0
        self.pheromone = [initial_tau] * self.num_items

        # heuristic: v/w
        self.heuristic = []
        for w, v in items:
            if w <= 0:
                self.heuristic.append(1e-6)
            else:
                self.heuristic.append(v / w)

        self.history = []

    def _get_weight_value(self, solution_indices):
        total_w, total_v = 0.0, 0.0
        for i in solution_indices:
            total_w += self.items[i][0]
            total_v += self.items[i][1]
        return total_w, total_v

    def _construct_ant_solution(self):
        current_weight = 0.0
        current_solution = []
        candidate_items = list(range(self.num_items))

        # We'll build probabilistically: pick next item among feasible ones weighted by tau^alpha * eta^beta
        while True:
            feasible = []
            scores = []
            for j in candidate_items:
                if current_weight + self.items[j][0] <= self.capacity:
                    feasible.append(j)
                    tau = (self.pheromone[j] ** self.alpha)
                    eta = (self.heuristic[j] ** self.beta)
                    scores.append(tau * eta)
            if not feasible:
                break
            # normalize probabilities
            s = sum(scores)
            if s <= 0:
                # pick random feasible
                chosen = random.choice(feasible)
            else:
                probs = [x / s for x in scores]
                idx = random.choices(range(len(feasible)), weights=probs, k=1)[0]
                chosen = feasible[idx]

            current_solution.append(chosen)
            current_weight += self.items[chosen][0]
            candidate_items.remove(chosen)
        return current_solution

    def run(self):
        best_global_value = 0.0
        best_global_solution = []
        best_global_weight = 0.0

        Q = 1.0
        tau_min = 1e-6
        tau_max = 1e6

        stagnation_counter = 0
        stagnation_threshold = max(10, int(self.max_iter * 0.15))

        for it in range(self.max_iter):
            iter_solutions = []
            iter_values = []

            for _ in range(self.num_ants):
                sol = self._construct_ant_solution()
                w, v = self._get_weight_value(sol)
                iter_solutions.append(sol)
                iter_values.append(v)
                if w <= self.capacity and v > best_global_value:
                    best_global_value = v
                    best_global_solution = sol[:]
                    best_global_weight = w

            # pheromone evaporation
            for i in range(self.num_items):
                self.pheromone[i] *= (1 - self.rho)

            # deposit: proportional to value (bigger value -> bigger deposit)
            # ensure we deposit more for better solutions
            for sol, val in zip(iter_solutions, iter_values):
                if val <= 0:
                    continue
                delta = val  # proportional deposit (you can scale by Q if needed)
                for idx in sol:
                    self.pheromone[idx] += delta / max(1.0, len(sol))  # normalize by solution length to avoid extremely big jumps

            # elite deposit (best-so-far)
            if best_global_value > 0 and best_global_solution:
                delta_best = best_global_value
                for idx in best_global_solution:
                    self.pheromone[idx] += delta_best / max(1.0, len(best_global_solution))

            # clamp pheromone
            for i in range(self.num_items):
                if self.pheromone[i] < tau_min:
                    self.pheromone[i] = tau_min
                if self.pheromone[i] > tau_max:
                    self.pheromone[i] = tau_max

            # stagnation detection
            # if no improvement in this iteration (best of iter equals previous best), increase counter
            if len(self.history) == 0 or best_global_value > (self.history[-1][1] if self.history else 0.0):
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_threshold:
                # partial reset: add small random noise or reduce pheromone bias
                for i in range(self.num_items):
                    self.pheromone[i] = self.pheromone[i] * 0.5 + random.random() * 0.1
                stagnation_counter = 0

            self.history.append((it + 1, best_global_value))
            self.progress_updated.emit(it + 1, best_global_value, best_global_weight)

        # finished
        self.finished.emit(best_global_solution, best_global_value, best_global_weight, self.history)

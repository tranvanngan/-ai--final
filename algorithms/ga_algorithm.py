"""
Genetic Algorithm (GA) implementation for Knapsack Problem
"""
import random
from PyQt6.QtCore import QThread, pyqtSignal


class GAWorker(QThread):
    progress_updated = pyqtSignal(int, float, float)
    finished = pyqtSignal(list, float, float, list)

    def __init__(self, items, capacity, pop_size, generations, mutation_rate, crossover_rate, penalty_factor):
        super().__init__()
        self.items = items
        self.capacity = capacity
        self.pop_size = max(4, pop_size)
        self.generations = max(1, generations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_items = len(items)
        self.penalty_factor = penalty_factor
        self.history = []

    def run(self):
        TOURNAMENT_SIZE = min(5, max(2, int(self.pop_size * 0.1)))
        ELITE_SIZE = max(1, int(self.pop_size * 0.05))

        population = self._initialize_population()
        # ensure initial population repaired
        for ind in population:
            self._repair_individual(ind)

        best_overall_solution = None
        best_overall_value = 0.0
        best_overall_weight = 0.0

        for gen in range(self.generations):
            # compute fitness (value for feasible solutions, penalized for infeasible)
            fitness_scores = [self._calculate_fitness(ind) for ind in population]

            # find best by fitness (but we want feasible best value preferentially)
            best_index = max(range(len(population)), key=lambda i: fitness_scores[i])
            best_candidate = population[best_index]
            w, v = self._get_weight_value(best_candidate)

            # If candidate feasible, treat value as improvement metric; otherwise still consider but less priority
            if w <= self.capacity and v > best_overall_value:
                best_overall_value = v
                best_overall_solution = best_candidate[:]
                best_overall_weight = w

            # fallback: if no feasible found yet, keep highest fitness (may be penalized)
            if best_overall_solution is None:
                # choose best feasible-like (largest fitness)
                idx = max(range(len(population)), key=lambda i: fitness_scores[i])
                w2, v2 = self._get_weight_value(population[idx])
                best_overall_solution = population[idx][:]
                best_overall_value = v2
                best_overall_weight = w2

            # record history (best overall value so far)
            self.history.append((gen + 1, best_overall_value))
            self.progress_updated.emit(gen + 1, best_overall_value, best_overall_weight)

            # sort population by fitness descending
            pop_with_scores = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            new_population = [ind[:] for ind, _ in pop_with_scores[:ELITE_SIZE]]  # elites copied

            # fill rest by selection + crossover + mutation
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_select(population, fitness_scores, TOURNAMENT_SIZE)
                parent2 = self._tournament_select(population, fitness_scores, TOURNAMENT_SIZE)
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                self._mutation(child1)
                self._mutation(child2)

                # repair children to be feasible (greedy remove low ratio items)
                self._repair_individual(child1)
                self._repair_individual(child2)

                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            population = new_population[:self.pop_size]

        # prepare output best solution indices
        if not best_overall_solution:
            self.finished.emit([], 0.0, 0.0, self.history)
            return

        selected_indices = [i for i, g in enumerate(best_overall_solution) if g == 1]
        self.finished.emit(selected_indices, best_overall_value, best_overall_weight, self.history)

    def _initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            # random but biased: include items with prob proportional to heuristic (v/w)
            individual = []
            for j in range(self.num_items):
                prob = min(0.9, (self.items[j][1] / max(1e-6, self.items[j][0])) / 10.0 + 0.1)
                individual.append(1 if random.random() < prob else 0)
            pop.append(individual)
        return pop

    def _get_weight_value(self, individual):
        total_weight, total_value = 0.0, 0.0
        if not individual or len(individual) != self.num_items:
            return 0.0, 0.0
        for i, gene in enumerate(individual):
            if gene == 1:
                total_weight += self.items[i][0]
                total_value += self.items[i][1]
        return total_weight, total_value

    def _calculate_fitness(self, individual):
        w, v = self._get_weight_value(individual)
        if w <= self.capacity:
            return v  # prefer feasible with raw value
        else:
            # penalize infeasible: larger overweight -> bigger penalty
            overweight = w - self.capacity
            # penalty_factor is a multiplier; smaller factor => softer penalty
            return max(0.0, v - self.penalty_factor * overweight)

    def _tournament_select(self, population, fitness_scores, k):
        k = min(k, len(population))
        participants = random.sample(range(len(population)), k)
        best = max(participants, key=lambda i: fitness_scores[i])
        return population[best][:]

    def _crossover(self, p1, p2):
        if self.num_items <= 1:
            return p1[:], p2[:]
        point = random.randint(1, self.num_items - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutation(self, individual):
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]

    def _repair_individual(self, individual):
        # If overweight, remove items with smallest value/weight (least efficient) until feasible
        w, v = self._get_weight_value(individual)
        if w <= self.capacity:
            return
        # build list of selected items sorted by ratio ascending (least useful first)
        selected = [(i, self.items[i][1] / max(1e-9, self.items[i][0])) for i, g in enumerate(individual) if g == 1]
        # sort by ratio ascending -> remove these first
        selected.sort(key=lambda x: x[1])
        for idx, _ in selected:
            if w <= self.capacity:
                break
            individual[idx] = 0
            w -= self.items[idx][0]

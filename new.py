# knapsack_ga_aco_fixed.py
import sys
import random
import math
import time

# Matplotlib + PyQt6
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFormLayout, QTextEdit, QSpinBox,
    QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# --- Mpl Canvas ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='lightgray')
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.axes.set_title("Biểu đồ hội tụ (Convergence Plot)", fontsize=10)
        self.axes.set_xlabel("Thế hệ / Vòng lặp", fontsize=8)
        self.axes.set_ylabel("Giá trị tốt nhất (Value)", fontsize=8)
        self.figure.tight_layout(pad=2)

    def clear_plot(self):
        self.axes.clear()
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.axes.set_title("Biểu đồ hội tụ (Convergence Plot)", fontsize=10)
        self.axes.set_xlabel("Thế hệ / Vòng lặp", fontsize=8)
        self.axes.set_ylabel("Giá trị tốt nhất (Value)", fontsize=8)
        self.draw()

# ---------------- GA Worker (improved) ----------------
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

# ---------------- ACO Worker (improved) ----------------
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

# ---------------- GUI (mostly same as your original, with small fixes) ----------------
class KnapsackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giải Knapsack: GA vs ACO (Fixed)")
        self.setGeometry(100, 100, 1200, 800)

        self.worker = None
        self.is_comparing = False
        self.comparison_results = {}
        self.aco_params_cache = None
        self.ga_params_cache = None
        self.start_time = 0

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 2)

        # problem params
        problem_group = QGroupBox("Thông số & Dữ liệu")
        problem_layout = QFormLayout()
        self.capacity_input = QLineEdit("500")
        self.num_items_input = QSpinBox()
        self.num_items_input.setRange(1, 1000)
        self.num_items_input.setValue(50)
        self.num_items_input.valueChanged.connect(self.update_table_rows)

        self.scenario_selector = QComboBox()
        self.scenario_selector.addItems([
            "Ngẫu nhiên (Uncorrelated)",
            "Tương quan (Weakly Correlated)",
            "Chống tương quan (Anti-Correlated)"
        ])

        problem_layout.addRow(QLabel("Sức chứa (Capacity):"), self.capacity_input)
        problem_layout.addRow(QLabel("Số lượng vật phẩm:"), self.num_items_input)
        problem_layout.addRow(QLabel("Loại Dữ liệu:"), self.scenario_selector)
        problem_group.setLayout(problem_layout)
        left_layout.addWidget(problem_group)

        algo_group = QGroupBox("Chọn chế độ")
        algo_layout = QFormLayout()
        self.algo_selector = QComboBox()
        self.algo_selector.addItems([
            "GA (Thuật toán Di truyền)",
            "ACO (Tối ưu hóa Đàn kiến)",
            "So sánh GA vs. ACO"
        ])
        self.algo_selector.currentTextChanged.connect(self.toggle_param_boxes)
        algo_layout.addRow(QLabel("Chế độ chạy:"), self.algo_selector)
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)

        params_layout = QHBoxLayout()
        self.ga_group = QGroupBox("Thông số GA")
        ga_layout = QFormLayout()
        self.pop_size_input = QLineEdit("100")
        self.generations_input = QLineEdit("200")
        self.penalty_factor_input = QLineEdit("1.0")
        self.mutation_rate_input = QLineEdit("0.01")
        self.crossover_rate_input = QLineEdit("0.9")
        ga_layout.addRow(QLabel("Pop Size:"), self.pop_size_input)
        ga_layout.addRow(QLabel("Generations:"), self.generations_input)
        ga_layout.addRow(QLabel("Penalty (P):"), self.penalty_factor_input)
        ga_layout.addRow(QLabel("Mut Rate:"), self.mutation_rate_input)
        ga_layout.addRow(QLabel("Cross Rate:"), self.crossover_rate_input)
        self.ga_group.setLayout(ga_layout)
        params_layout.addWidget(self.ga_group)

        self.aco_group = QGroupBox("Thông số ACO")
        aco_layout = QFormLayout()
        self.ants_input = QLineEdit("50")
        self.aco_iter_input = QLineEdit("200")
        self.alpha_input = QLineEdit("1.0")
        self.beta_input = QLineEdit("2.0")
        self.rho_input = QLineEdit("0.1")
        aco_layout.addRow(QLabel("Ants:"), self.ants_input)
        aco_layout.addRow(QLabel("Iterations:"), self.aco_iter_input)
        aco_layout.addRow(QLabel("Alpha (Pheromone):"), self.alpha_input)
        aco_layout.addRow(QLabel("Beta (Heuristic):"), self.beta_input)
        aco_layout.addRow(QLabel("Rho (Evaporation):"), self.rho_input)
        self.aco_group.setLayout(aco_layout)
        params_layout.addWidget(self.aco_group)
        left_layout.addLayout(params_layout)

        control_layout = QHBoxLayout()
        self.random_data_btn = QPushButton("Tạo dữ liệu theo Scenario")
        self.random_data_btn.clicked.connect(self.populate_random_data)
        self.run_btn = QPushButton("Bắt đầu giải")
        self.run_btn.clicked.connect(self.run_solve)
        control_layout.addWidget(self.random_data_btn)
        control_layout.addWidget(self.run_btn)
        left_layout.addLayout(control_layout)

        self.items_table = QTableWidget()
        self.items_table.setColumnCount(3)
        self.items_table.setHorizontalHeaderLabels(["Vật phẩm #", "Trọng lượng (W)", "Giá trị (V)"])
        self.items_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(self.items_table)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 1)

        self.plot_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        right_layout.addWidget(QLabel("Biểu đồ Hội tụ:"))
        right_layout.addWidget(self.plot_canvas)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Log tiến trình...")
        right_layout.addWidget(QLabel("Tiến trình:"))
        right_layout.addWidget(self.log_output)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("Kết quả cuối cùng...")
        right_layout.addWidget(QLabel("Kết quả:"))
        right_layout.addWidget(self.result_output)

        # init
        self.update_table_rows()
        self.populate_random_data()
        self.toggle_param_boxes(self.algo_selector.currentText())

    def toggle_param_boxes(self, text):
        if "So sánh" in text:
            self.ga_group.show()
            self.aco_group.show()
        elif "GA" in text:
            self.ga_group.show()
            self.aco_group.hide()
        elif "ACO" in text:
            self.ga_group.hide()
            self.aco_group.show()

    def update_table_rows(self):
        self.items_table.setRowCount(self.num_items_input.value())
        for row in range(self.items_table.rowCount()):
            if self.items_table.item(row, 0) is None:
                item_num = QTableWidgetItem(f"{row + 1}")
                item_num.setFlags(item_num.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.items_table.setItem(row, 0, item_num)

    def populate_random_data(self):
        try:
            num_items = self.items_table.rowCount()
            scenario_type = self.scenario_selector.currentText()

            MIN_W, MAX_W = 1, 100
            MIN_V, MAX_V_UNCORRELATED = 1, 150

            weights = [random.randint(MIN_W, MAX_W) for _ in range(num_items)]

            values = []
            if "Ngẫu nhiên" in scenario_type:
                values = [random.randint(MIN_V, MAX_V_UNCORRELATED) for _ in range(num_items)]
            elif "Tương quan" in scenario_type:
                values = [w + random.randint(1, 10) for w in weights]
            elif "Chống tương quan" in scenario_type:
                values = [MAX_W - w + random.randint(1, 10) for w in weights]

            total_weight = sum(weights)
            self.capacity_input.setText(str(int(total_weight * 0.5)))

            max_v = max(values) if values else 1
            max_w = max(weights) if weights else 1
            penalty_suggest = (max_v / max_w) if max_w != 0 else 1.0
            self.penalty_factor_input.setText(f"{penalty_suggest:.2f}")

            for row in range(num_items):
                self.items_table.setItem(row, 1, QTableWidgetItem(str(weights[row])))
                self.items_table.setItem(row, 2, QTableWidgetItem(str(values[row])))

            self.log_output.append(f"Đã tạo {num_items} vật phẩm.")
            self.log_output.append(f"Loại: {scenario_type}.")
            self.log_output.append(f"Tổng trọng lượng: {total_weight}.")
            self.log_output.append(f"Sức chứa đề xuất: {self.capacity_input.text()}. P_GA đề xuất: {self.penalty_factor_input.text()}")
            self.log_output.append("Vui lòng chỉnh lại Capacity để thử các kịch bản Ràng buộc Lỏng/Chặt.")

        except ValueError:
            self.log_output.append("Lỗi: Vui lòng nhập số hợp lệ.")

    def run_solve(self):
        if self.worker and self.worker.isRunning():
            return

        self.log_output.clear()
        self.result_output.clear()
        self.plot_canvas.clear_plot()
        self.is_comparing = False
        self.comparison_results = {}

        try:
            capacity = float(self.capacity_input.text())
            items = []
            for row in range(self.items_table.rowCount()):
                w_item = self.items_table.item(row, 1)
                v_item = self.items_table.item(row, 2)
                if not w_item or not v_item or not w_item.text() or not v_item.text():
                    raise ValueError(f"Thiếu dữ liệu dòng {row + 1}")
                items.append((float(w_item.text()), float(v_item.text())))

            if any(w <= 0 for w, v in items):
                self.log_output.append("Lỗi: Trọng lượng vật phẩm phải lớn hơn 0.")
                return

            algo_choice = self.algo_selector.currentText()

            # GA params
            ga_params = (
                items, capacity,
                int(self.pop_size_input.text()), int(self.generations_input.text()),
                float(self.mutation_rate_input.text()), float(self.crossover_rate_input.text()),
                float(self.penalty_factor_input.text())
            )

            # ACO params
            aco_params = (
                items, capacity,
                int(self.ants_input.text()), int(self.aco_iter_input.text()),
                float(self.alpha_input.text()), float(self.beta_input.text()), float(self.rho_input.text())
            )

            if "So sánh" in algo_choice:
                self.is_comparing = True
                self.log_output.append("--- BẮT ĐẦU SO SÁNH ---")
                self.aco_params_cache = aco_params
                self.ga_params_cache = ga_params
                self.log_output.append(f"1. Đang chạy GA (Gens: {ga_params[3]}, P: {ga_params[-1]:.2f})...")
                self.worker = GAWorker(*self.ga_params_cache)

            elif "GA" in algo_choice:
                self.log_output.append(f"Đang chạy GA (Gens: {ga_params[3]}, P: {ga_params[-1]:.2f})...")
                self.worker = GAWorker(*ga_params)

            elif "ACO" in algo_choice:
                self.log_output.append(f"Đang chạy ACO (Iters: {aco_params[3]}, Alpha/Beta: {aco_params[4]}/{aco_params[5]})...")
                self.worker = ACOWorker(*aco_params)

            self.run_btn.setEnabled(False)
            self.run_btn.setText("Đang chạy...")
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.finished.connect(self.handle_worker_finished)

            self.start_time = time.time()
            self.worker.start()

        except Exception as e:
            self.log_output.append(f"Lỗi Input: {e}")
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Bắt đầu giải")
            self.is_comparing = False

    def handle_worker_finished(self, selected_indices, final_val, final_weight, history):
        self.show_results(selected_indices, final_val, final_weight, history)

    def update_progress(self, iteration, best_val, best_w):
        algo_name = "ACO" if isinstance(self.worker, ACOWorker) else "GA"
        self.log_output.append(f"[{algo_name}] Iter/Gen {iteration} | Best value so far: {best_val:.2f} | Weight: {best_w:.2f}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def show_results(self, selected_indices, final_val, final_weight, history):
        duration = time.time() - self.start_time
        algo_name = "ACO" if isinstance(self.worker, ACOWorker) else "GA"

        if self.is_comparing:
            if "GA" not in self.comparison_results:
                self.log_output.append(f"-> GA hoàn thành trong {duration:.3f}s\n")
                self.comparison_results["GA"] = {
                    "val": final_val, "w": final_weight, "time": duration,
                    "items": selected_indices, "history": history
                }
                self.log_output.append(f"2. Đang chạy ACO (Iters: {self.aco_params_cache[3]}, Alpha/Beta: {self.aco_params_cache[4]}/{self.aco_params_cache[5]})...")
                self.worker = ACOWorker(*self.aco_params_cache)
                self.worker.progress_updated.connect(self.update_progress)
                self.worker.finished.connect(self.handle_worker_finished)
                self.start_time = time.time()
                self.worker.start()
                return
            else:
                self.log_output.append(f"-> ACO hoàn thành trong {duration:.3f}s\n")
                self.comparison_results["ACO"] = {
                    "val": final_val, "w": final_weight, "time": duration,
                    "items": selected_indices, "history": history
                }
                self.display_comparison_results()
                self.plot_convergence(self.comparison_results["GA"]["history"], self.comparison_results["ACO"]["history"])
                self.is_comparing = False
                self.run_btn.setEnabled(True)
                self.run_btn.setText("Bắt đầu giải")
                return

        # single run
        self.log_output.append(f"--- Hoàn thành trong {duration:.3f}s ---")
        self.result_output.append("--- KẾT QUẢ ĐƠN LẺ ---")
        self.result_output.append(f"Thuật toán: {self.algo_selector.currentText()}")
        self.result_output.append(f"Giá trị: {final_val:.2f}")
        self.result_output.append(f"Trọng lượng: {final_weight:.2f} / {self.capacity_input.text()}")

        items_str = [f"#{i + 1}" for i in selected_indices]
        items_str.sort(key=lambda x: int(x[1:]))
        self.result_output.append(f"Vật phẩm: {', '.join(items_str)}")

        # plot
        self.plot_convergence(history, history_aco=None, label=algo_name)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("Bắt đầu giải")

    def plot_convergence(self, history_ga, history_aco=None, label="GA"):
        self.plot_canvas.clear_plot()
        ax = self.plot_canvas.axes

        if history_ga:
            gens = [h[0] for h in history_ga]
            vals = [h[1] for h in history_ga]
            ax.plot(gens, vals, label=label, marker='.', markersize=4, linestyle='-')

        if history_aco:
            its = [h[0] for h in history_aco]
            vals2 = [h[1] for h in history_aco]
            ax.plot(its, vals2, label="ACO", marker='x', markersize=4, linestyle='--')

        ax.legend(fontsize=8)
        ax.set_title("Biểu đồ hội tụ: Giá trị tốt nhất theo thời gian", fontsize=10)
        ax.set_xlabel("Thế hệ / Vòng lặp", fontsize=8)
        ax.set_ylabel("Giá trị tốt nhất (Value)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        self.plot_canvas.draw()

    def display_comparison_results(self):
        ga = self.comparison_results.get("GA", {})
        aco = self.comparison_results.get("ACO", {})
        if not ga or not aco:
            self.result_output.append("Lỗi: Không đủ dữ liệu so sánh.")
            return

        self.result_output.clear()
        self.result_output.append("=== BẢNG SO SÁNH GA vs ACO ===")
        self.result_output.append("\n[Thuật toán Di truyền (GA)]")
        self.result_output.append(f"Tham số: Pop={self.ga_params_cache[2]}, Gen={self.ga_params_cache[3]}, P={self.ga_params_cache[-1]:.2f}")
        self.result_output.append(f"Giá trị: {ga['val']:.2f}")
        self.result_output.append(f"Trọng lượng: {ga['w']:.2f} / {self.capacity_input.text()}")
        self.result_output.append(f"Thời gian: {ga['time']:.3f}s")

        self.result_output.append("\n[Tối ưu hóa Đàn kiến (ACO)]")
        self.result_output.append(f"Tham số: Ants={self.aco_params_cache[2]}, Iters={self.aco_params_cache[3]}, A/B/R={self.aco_params_cache[4]}/{self.aco_params_cache[5]}/{self.aco_params_cache[6]}")
        self.result_output.append(f"Giá trị: {aco['val']:.2f}")
        self.result_output.append(f"Trọng lượng: {aco['w']:.2f} / {self.capacity_input.text()}")
        self.result_output.append(f"Thời gian: {aco['time']:.3f}s")

        self.result_output.append("\n--- ĐÁNH GIÁ TỔNG QUAN ---")
        if ga['val'] > aco['val']:
            self.result_output.append(f"🏆 Giá trị tối ưu: GA thắng! (Tốt hơn {ga['val'] - aco['val']:.2f})")
        elif aco['val'] > ga['val']:
            self.result_output.append(f"🏆 Giá trị tối ưu: ACO thắng! (Tốt hơn {aco['val'] - ga['val']:.2f})")
        else:
            self.result_output.append("🤝 Giá trị tối ưu: Hòa nhau.")

        if ga['time'] < aco['time']:
            self.result_output.append(f"⚡ Tốc độ: GA nhanh hơn! (Nhanh hơn {aco['time'] - ga['time']:.3f}s)")
        else:
            self.result_output.append(f"⚡ Tốc độ: ACO nhanh hơn! (Nhanh hơn {ga['time'] - aco['time']:.3f}s)")

        items_ga_str = [f"#{i + 1}" for i in ga['items']]
        items_ga_str.sort(key=lambda x: int(x[1:]))
        self.result_output.append(f"\n[GA Items: {', '.join(items_ga_str)}]")
        items_aco_str = [f"#{i + 1}" for i in aco['items']]
        items_aco_str.sort(key=lambda x: int(x[1:]))
        self.result_output.append(f"[ACO Items: {', '.join(items_aco_str)}]")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    app = QApplication(sys.argv)
    window = KnapsackApp()
    window.show()
    sys.exit(app.exec())

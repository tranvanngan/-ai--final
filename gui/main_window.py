"""
Main window GUI for Knapsack Problem Solver
"""
import random
import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFormLayout, QTextEdit, QSpinBox,
    QComboBox
)
from PyQt6.QtCore import Qt

from algorithms.ga_algorithm import GAWorker
from algorithms.aco_algorithm import ACOWorker
from gui.canvas import MplCanvas


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

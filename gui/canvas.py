"""
Matplotlib Canvas for plotting convergence graphs
"""
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

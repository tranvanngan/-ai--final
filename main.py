"""
Main entry point for Knapsack Problem Solver
GA vs ACO Comparison Application
"""
import sys
import warnings
from PyQt6.QtWidgets import QApplication
from gui.main_window import KnapsackApp


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    app = QApplication(sys.argv)
    window = KnapsackApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

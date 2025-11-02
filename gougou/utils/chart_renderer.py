import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from typing import Optional, Tuple

# Use non-interactive backend
matplotlib.use('Qt5Agg')

class MatplotlibCanvas(QWidget):
    """Matplotlib canvas widget for PyQt6"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def clear(self):
        """Clear the figure"""
        self.figure.clear()
        self.canvas.draw()
    
    def draw(self):
        """Redraw the canvas"""
        self.canvas.draw()

def plot_probability_distribution(ax, proba: np.ndarray, y_true: Optional[np.ndarray] = None):
    """Plot probability distribution histogram"""
    ax.clear()
    
    if y_true is not None:
        # Separate by class
        proba_0 = proba[y_true == 0]
        proba_1 = proba[y_true == 1]
        ax.hist(proba_0, bins=30, alpha=0.6, label='Non-Default', color='blue', edgecolor='black')
        ax.hist(proba_1, bins=30, alpha=0.6, label='Default', color='red', edgecolor='black')
        ax.legend()
    else:
        ax.hist(proba, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    ax.set_xlabel('Probability of Default')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution')
    ax.grid(True, alpha=0.3)

def plot_gains_curve(ax, y_true: np.ndarray, proba: np.ndarray):
    """Plot gains curve (cumulative capture)"""
    ax.clear()
    
    # Sort by probability descending
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = np.sum(y_true)
    
    n = len(y_true)
    x = np.arange(0, n+1) / n * 100
    y = np.concatenate([[0], cum_pos]) / max(total_pos, 1) * 100
    
    # Plot gains curve
    ax.plot(x, y, 'b-', linewidth=2, label='Model')
    
    # Plot baseline (random)
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Random')
    
    # Plot perfect model
    ax.plot([0, total_pos/n*100, 100], [0, 100, 100], 'g--', linewidth=1.5, label='Perfect')
    
    ax.set_xlabel('% of Population')
    ax.set_ylabel('% of Defaults Captured')
    ax.set_title('Gains Curve (Cumulative Capture)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

def plot_calibration(ax, y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    """Plot calibration curve"""
    ax.clear()
    
    # Bin probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_pos_rate = []
    mean_proba = []
    
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i+1])
        if mask.sum() > 0:
            true_pos_rate.append(y_true[mask].mean())
            mean_proba.append(proba[mask].mean())
    
    if true_pos_rate:
        ax.plot(mean_proba, true_pos_rate, 'bo-', linewidth=2, markersize=6, label='Model')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Actual Positive Rate')
        ax.set_title('Calibration Plot (Reliability)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

def plot_confusion_matrix(ax, tp: int, fp: int, fn: int, tn: int):
    """Plot confusion matrix heatmap"""
    ax.clear()
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_title('Confusion Matrix @ Cut-off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)

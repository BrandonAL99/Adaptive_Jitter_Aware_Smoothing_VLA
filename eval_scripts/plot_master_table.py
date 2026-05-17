#!/usr/bin/env python3
"""
GUI for plotting metric comparison charts from Thesis Results Final.xlsx.

Reads raw episode data directly from each sheet (no formula caching needed).
Run from repo root:
    python tools/plot_master_table.py
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import openpyxl
from pathlib import Path
from tkinter import ttk, filedialog

EXCEL_PATH = Path.home() / "Documents/sem_1_2026/thesis/Thesis Results Final v2.xlsx"
#EXCEL_PATH = Path.home() / "Documents/sem_1_2026/thesis/Thesis_results_successonly.xlsx"

SUITES = ["Spatial", "Object", "Goal", "Long"]
ALL_SUITES = SUITES + ["Average"]

# (display_label, sheet_name, (data_row_start, data_row_end))
# sheet_name=None → hardcoded paper values only
METHODS = [
    ("SmolVLA [paper]",    None,                  None),
    ("Baseline_exec1",     "Baseline_exec1",       (4, 103)),
    ("Libero_eval_chunk50","Libero_eval_chunk50",  (4, 103)),
    ("Libero_LP_2Hz",      "Libero_LP_2Hz",        (4, 103)),
    ("Libero_LP_4Hz",      "Libero_LP_4Hz",        (4, 103)),
    ("Libero_LP_6Hz",      "Libero_LP_6Hz",        (4, 103)),
    ("AJAS_Linear",        "AJAS_Linear",          (4,  33)),
    ("AJAS_Spline",        "AJAS_Spline",          (4, 103)),
    ("AJAS_Spline_old",    "AJAS_Spline_old",      (4, 103)),
]

# 1-indexed column numbers per suite
# Block offsets: Spatial=col2, Object=col12, Goal=col22, Long=col32
# Within block: success+1, sr+2, sm_raw+3, sm_proc+4, ser+5, jerk+6, dcc+7
SUITE_COLS = {
    "Spatial": {"success": 3,  "sm": 6,  "ser": 7,  "jerk": 8,  "dcc": 9},
    "Object":  {"success": 13, "sm": 16, "ser": 17, "jerk": 18, "dcc": 19},
    "Goal":    {"success": 23, "sm": 26, "ser": 27, "jerk": 28, "dcc": 29},
    "Long":    {"success": 33, "sm": 36, "ser": 37, "jerk": 38, "dcc": 39},
}

# SmolVLA paper SR (hardcoded from paper, no other metrics available)
PAPER_SR = {"Spatial": 90, "Object": 96, "Goal": 92, "Long": 71}

# Multiplier cells in MasterTableV2 (val_cell, std_cell) — None = multiplier of 1
MULTIPLIER_CELLS = {
    "SR":   (None,  None),
    "Sm":   ("E17", "E17"),
    "SER":  ("E28", "E28"),
    "Jerk": ("E39", "E39"),
    "DCC":  ("E50", "E50"),
}

COLORS = [
    "#9467BD",  # purple - SmolVLA paper
    "#FF69B4",  # pink - Baseline_exec1
    "#8C8C8C",  # grey - Libero_eval_chunk50
    "#C44E52",  # red - Libero_LP_2Hz
    "#E69F00",  # orange - Libero_LP_4Hz
    "#F0E442",  # yellow - Libero_LP_6Hz
    "#4C72B0",  # blue - AJAS_Linear
    "#2CA02C",  # dark green - AJAS_Spline
    "#55A868",  # green - AJAS_Spline_old
    
]


# ── data loading ──────────────────────────────────────────────────────────────

_wb = None

def get_wb():
    global _wb
    if _wb is None:
        _wb = openpyxl.load_workbook(str(EXCEL_PATH), data_only=True)
    return _wb


def get_multipliers(metric):
    val_cell, std_cell = MULTIPLIER_CELLS[metric]
    if val_cell is None:
        return 1.0, 1.0
    mv = get_wb()["MasterTableV2"]
    v = mv[val_cell].value
    s = mv[std_cell].value
    return float(v) if v else 1.0, float(s) if s else 1.0


def read_numeric_col(ws, col, row_start, row_end):
    vals = []
    for r in range(row_start, row_end + 1):
        v = ws.cell(row=r, column=col).value
        if isinstance(v, bool):
            vals.append(float(v))
        elif isinstance(v, (int, float)):
            vals.append(float(v))
    return np.array(vals, dtype=float)


def compute_metric(metric, active_labels=None):
    """
    Returns means, stds — each a dict: {method_label: [spatial, object, goal, long, average]}
    NaN where data unavailable. active_labels filters which methods to include.
    """
    wb = get_wb()
    val_mult, std_mult = get_multipliers(metric)
    means, stds = {}, {}

    for label, sheet_name, data_rows in METHODS:
        if active_labels is not None and label not in active_labels:
            continue
        sm, ss = [], []

        if sheet_name is None:
            for suite in SUITES:
                sm.append(PAPER_SR[suite] if metric == "SR" else np.nan)
                ss.append(np.nan)
            avg = float(np.nanmean(sm)) if any(not np.isnan(v) for v in sm) else np.nan
            sm.append(avg); ss.append(np.nan)
            means[label], stds[label] = sm, ss
            continue

        ws = wb[sheet_name]
        r0, r1 = data_rows

        for suite in SUITES:
            if metric == "SR":
                vals = read_numeric_col(ws, SUITE_COLS[suite]["success"], r0, r1)
                n = r1 - r0 + 1
                sm.append(100.0 * vals.sum() / n if len(vals) > 0 else np.nan)
                ss.append(np.nan)
            else:
                col_key = metric.lower()
                if col_key not in SUITE_COLS[suite]:
                    sm.append(np.nan); ss.append(np.nan); continue
                col = SUITE_COLS[suite][col_key]
                vals = read_numeric_col(ws, col, r0, r1)
                if len(vals) > 0:
                    sm.append(float(np.mean(vals)) * val_mult)
                    ss.append(float(np.std(vals, ddof=1)) * std_mult)
                else:
                    sm.append(np.nan); ss.append(np.nan)

        avg_m = float(np.nanmean(sm)) if any(not np.isnan(v) for v in sm) else np.nan
        avg_s = float(np.nanmean(ss)) if any(not np.isnan(v) for v in ss) else np.nan
        sm.append(avg_m); ss.append(avg_s)
        means[label], stds[label] = sm, ss

    return means, stds


# ── chart ─────────────────────────────────────────────────────────────────────

METRIC_LABELS = { 
    "SR":   "Success Rate (%)",
    "Sm":   "Sm (×1000)",
    "SER":  "Sm reduction (Sm_proc / Sm_raw)",
    "Jerk": "Normalised Jerk",
    "DCC":  "DCC (direction change count)",
}

X_LABELS = SUITES + ["Average"]

def draw_chart(fig, metric, active_labels, active_suites):
    fig.clear()
    ax = fig.add_subplot(111)

    means, stds = compute_metric(metric, active_labels)
    val_mult, _ = get_multipliers(metric)

    # Filter out "Average" from active_suites
    #active_suites = [s for s in active_suites if s != "Average"]
    
    n_suites = len(active_suites)
    active_methods = [(label, ci) for ci, (label, _, _) in enumerate(METHODS)
                      if label in active_labels]
    n_methods = len(active_methods)

    # If only one suite selected, use method names as x-labels
    if False:#n_suites == 1:
        suite = active_suites[0]
        suite_idx = SUITES.index(suite)
        x_labels = [label for label, _ in active_methods]
        x = np.arange(n_methods)
        bar_w = 0.6
        
        for plot_i, (label, color_i) in enumerate(active_methods):
            m = means[label][suite_idx]
            s = stds[label][suite_idx]
            
            height = m if not np.isnan(m) else 0.0
            yerr = s if (not np.isnan(s) and metric != "SR") else None
            
            ax.bar(
                plot_i, height,
                width=bar_w,
                color=COLORS[color_i % len(COLORS)],
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 1.2, "alpha": 0.7},
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=14, rotation=45, ha='right')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
        
        mult_note = f"  (×{int(val_mult)})" if val_mult != 1.0 else ""
        ax.set_title(f"{metric} - {suite}{mult_note}", fontsize=18, fontweight="bold")
        
    else:
        # Multiple suites: grouped bars with legend
        x_labels = active_suites
        n_groups = len(x_labels)
        bar_w = 0.8 / max(n_methods, 1)
        x = np.arange(n_groups)

        for plot_i, (label, color_i) in enumerate(active_methods):
            m = means[label]
            s = stds[label]
            offsets = (plot_i - n_methods / 2 + 0.5) * bar_w

            filtered_m = []
            filtered_s = []

            for s_name in active_suites:
                idx = SUITES.index(s_name)
                filtered_m.append(m[idx])
                filtered_s.append(s[idx])

            heights = [v if not np.isnan(v) else 0.0 for v in filtered_m]
            errs = [v if (not np.isnan(v) and metric != "SR") else None for v in filtered_s]

            yerr = errs if any(e is not None for e in errs) else None
            if yerr:
                yerr = [e if e is not None else 0 for e in errs]

            ax.bar(
                x + offsets, heights,
                width=bar_w * 0.9,
                color=COLORS[color_i % len(COLORS)],
                label=label,
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 1.2, "alpha": 0.7},
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)

        mult_note = f"  (×{int(val_mult)})" if val_mult != 1.0 else ""
        ax.set_title(f"{metric}{mult_note}", fontsize=18, fontweight="bold")

        # Legend on right side for multiple suites
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize=12, framealpha=0.9)
    
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    
    # Increase tick label size
    ax.tick_params(axis='y', labelsize=14)
    
    fig.tight_layout()


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thesis Metrics — MasterTableV2")
        self.geometry("1100x680")
        self.resizable(True, True)

        self._suite_vars = {suite: tk.BooleanVar(value=True) for suite in ALL_SUITES}
        self._current_metric = tk.StringVar(value="SR")
        self._check_vars = {label: tk.BooleanVar(value=True) for label, _, _ in METHODS}
        self._build_ui()
        self._plot("SR")
    
    def _save_pdf(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"{self._current_metric.get()}_plot.pdf",
            title="Save plot as PDF",
        )

        if not file_path:
            return

        try:
            self.fig.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
            self._status.set(f"Saved PDF: {file_path}")
        except Exception as e:
            self._status.set(f"Failed to save PDF: {e}")

    def _active_suites(self):
        selected = [s for s, var in self._suite_vars.items() if var.get()]
        return selected
        
    def _active_labels(self):
        return {label for label, var in self._check_vars.items() if var.get()}

    def _build_ui(self):
        # Top button bar
        btn_frame = ttk.Frame(self, padding=6)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(btn_frame, text="Metric:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(4, 8))

        for metric in ["SR", "Sm", "SER", "Jerk", "DCC"]:
            ttk.Button(
                btn_frame, text=metric, width=8,
                command=lambda m=metric: self._plot(m),
            ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            btn_frame,
            text="Save PDF",
            command=self._save_pdf,
        ).pack(side=tk.RIGHT, padx=6)

        # Checkbox row
        cb_frame = ttk.LabelFrame(self, text="Methods", padding=(6, 2))
        cb_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        for i, (label, _, _) in enumerate(METHODS):
            color = COLORS[i % len(COLORS)]
            cb = tk.Checkbutton(
                cb_frame,
                text=label,
                variable=self._check_vars[label],
                command=self._replot,
                fg=color,
                selectcolor="white",
                font=("", 9),
            )
            cb.pack(side=tk.LEFT, padx=6)

        # Suite checkbox row
        suite_frame = ttk.LabelFrame(self, text="Suites", padding=(6, 2))
        suite_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        for suite in ALL_SUITES:
            cb = tk.Checkbutton(
                suite_frame,
                text=suite,
                variable=self._suite_vars[suite],
                command=self._replot,
                font=("", 9),
            )
            cb.pack(side=tk.LEFT, padx=6)

        # Status bar
        self._status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self._status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(side=tk.BOTTOM, fill=tk.X)

        # Matplotlib canvas
        self.fig = plt.Figure(figsize=(10, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar.update()

    def _replot(self):
        self._plot(self._current_metric.get())

    def _plot(self, metric):
        self._current_metric.set(metric)
        self._status.set(f"Loading {metric}…")
        self.update_idletasks()
        try:
            draw_chart(
                self.fig,
                metric,
                self._active_labels(),
                self._active_suites()
            )
            self.canvas.draw()
            val_mult, _ = get_multipliers(metric)
            mult_str = f"  (multiplier ×{int(val_mult)})" if val_mult != 1.0 else ""
            self._status.set(f"Showing: {metric}{mult_str}")
        except Exception as e:
            self._status.set(f"Error: {e}")
            raise


if __name__ == "__main__":
    app = App()
    app.mainloop()
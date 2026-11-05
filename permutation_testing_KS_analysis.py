# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:45:21 2026

"""
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from matplotlib.backends.backend_pdf import PdfPages
import os


# Output directory
save_dir = r"OutDir"
os.makedirs(save_dir, exist_ok=True)

# Output paths for PDF plots
pdf_all_path = os.path.join(save_dir, "ALL_KS_PLOTS.pdf")

# Base directory containing the confounder folders
base_dir = r"BaseDir"

# Provide a list of folders to contain outputs - each associated with a single confounder
confounder_folders = [
    "01",
    "02",
    "03",
    "04",
    "05"
]

okabe_ito = [
    (0/255, 114/255, 178/255),   # Blue
    (213/255, 94/255, 0/255),    # Vermillion
    (86/255, 180/255, 233/255),  # Sky Blue
    (0/255, 158/255, 115/255),   # Bluish Green
    (240/255, 228/255, 66/255),  # Yellow
    (204/255, 121/255, 167/255), # Reddish Purple
    (0/255, 0/255, 0/255),       # Black
]

# Assign colours explicitly 
colour_map = {}
for i, folder in enumerate(confounder_folders):
    colour_map[folder] = okabe_ito[i % len(okabe_ito)]

# Observed distribution colour (Okabe–Ito orange)
observed_colour = (230/255, 159/255, 0/255)

#Open PDFs
pdf_all = PdfPages(pdf_all_path)

for folder in confounder_folders:

    # Load data
    null_path = os.path.join(base_dir, folder, "outputs", "perm_null_full.csv")
    obs_path  = os.path.join(base_dir, folder, "outputs", "perm_test_results.csv")

    df_null = pd.read_csv(null_path)
    df_obs  = pd.read_csv(obs_path)

    null_vals = df_null["perm_null"].values
    obs_vals  = df_obs["auc_obs"].values

    # Two sample KS test
    if len(obs_vals) > 1:
        ks_stat, ks_p = ks_2samp(obs_vals, null_vals)
    else:
        ks_stat, ks_p = np.nan, np.nan

    alpha = 0.001 #Baseline significant for test - change as required
    significant = (ks_p < alpha)

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Histogram + KDE
    ax = axes[0]
    sns.histplot(null_vals, bins=30, stat="density",
                 color=colour_map[folder], alpha=0.5, label="Null", ax=ax)
    sns.kdeplot(null_vals, color=colour_map[folder], linewidth=2, ax=ax)

    sns.histplot(obs_vals, bins=15, stat="density",
                 color=observed_colour, alpha=0.5, label="Observed", ax=ax)
    if len(obs_vals) > 1:
        sns.kdeplot(obs_vals, color=observed_colour, linewidth=2, ax=ax)

    ax.set_title(f"{folder}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Density")
    ax.legend()

    # Subplot label (a)
    ax.text(-0.1, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Right: ECDF
    ax2 = axes[1]
    x_null = np.sort(null_vals)
    y_null = np.arange(1, len(x_null)+1) / len(x_null)
    ax2.step(x_null, y_null, color=colour_map[folder], label="Null ECDF")

    x_obs = np.sort(obs_vals)
    y_obs = np.arange(1, len(x_obs)+1) / len(x_obs)
    ax2.step(x_obs, y_obs, color=observed_colour, label="Observed ECDF")

    ax2.text(
        0.05, 0.95,
        f"KS statistic = {ks_stat:.6f}\nKS p-value = {ks_p:.10e}",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    ax2.text(
        0.05, 0.75,
        f"{'Significant' if significant else 'Not significant'} at α = 0.001",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    ax2.set_xlabel("AUC")
    ax2.set_ylabel("ECDF")
    ax2.legend()

    # Subplot label (b)
    ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    # Remove top and right spines
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major",
                       length=4, width=1, color="black", direction="out")
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    # Save to PDF
    pdf_all.savefig(fig)

    plt.close(fig)

# Close PDF
pdf_all.close()

print("Saved ALL_KS_PLOTS.pdf")

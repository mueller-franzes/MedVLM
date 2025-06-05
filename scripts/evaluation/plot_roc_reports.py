import pandas as pd
import ast
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

path_run = Path("/home/ve001107/MedVLM/runs/CTRATE/MedVLM_2025_05_25_150747_trainable/best-epoch=20-val/results_report/predictions_t2i.csv")
path_out = path_run.parent
df = pd.read_csv(path_run)

all_labels = []
all_scores = []

for _, row in df.iterrows():
    gt_list = np.fromstring(row["GT"][1:-1], sep=' ', dtype=int)  # remove brackets
    prob_list = np.fromstring(row["Prob"][1:-1], sep=' ', dtype=float)

    all_labels.extend(gt_list)
    all_scores.extend(prob_list)

# Calculate ROC AUC
auc = roc_auc_score(all_labels, all_scores)
print(f"ROC AUC: {auc:.3f}")

# Get points for the ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_scores)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Save the figure
plt.savefig(path_out/f'roc_t2i.png', dpi=300)
plt.close()  # Close the figure
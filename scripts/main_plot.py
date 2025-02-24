from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 

from medvlm.utils.roc_curve import plot_roc_curve, cm2acc, cm2x

path_out = Path.cwd()/'results'

label = "Emphysema"
df = pd.read_csv(Path.cwd()/f'predictions_{label}.csv')
# df = pd.read_csv(f'runs/CTRATE/MedVLM_2025_01_24_140928_w_contrastive/results/predictions_{label}.csv')

fontdict = {'fontsize': 11, 'fontweight': 'bold'}

#  -------------------------- Confusion Matrix -------------------------
cm = confusion_matrix(df['GT'], df['Pred'])
print(cm)


# ------------------------------- ROC-AUC ---------------------------------
fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(4,4)) 
y_pred_lab = np.asarray(df['Prob'])
y_true_lab = np.asarray(df['GT'])

tprs, fprs, auc_val, thrs, opt_idx, _ = plot_roc_curve(y_true_lab, y_pred_lab, axis, name=f'{label} AUC:', fontdict=fontdict)
axis.set_title(f'{label}', fontdict=fontdict)
fig.tight_layout()
fig.savefig(path_out/f'roc.png', dpi=300)


#  -------------------------- Confusion Matrix -------------------------
labels = ["No", "Yes"]
df_cm = pd.DataFrame(data=cm, columns=labels, index=labels)
fig, axis = plt.subplots(1, 1, figsize=(4,4))
sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True) 
axis.set_title(f'Confusion Matrix', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
rotation='vertical'
axis.set_title(f'{label}', fontdict=fontdict)
axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
axis.set_xlabel('Prediction' , fontdict=fontdict)
axis.set_ylabel('True' , fontdict=fontdict)
fig.tight_layout()
fig.savefig(path_out/f'confusion_matrix.png', dpi=300)
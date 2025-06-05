from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 

from medvlm.utils.roc_curve import plot_roc_curve

from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D

# ------------------------------- Settings ---------------------------------
path_run_dir = Path('/home/ve001107/MedVLM/runs/CTRATE/MedVLM_2025_05_28_040836_trainable/best-epoch=36-val/')
# path_run_dir = Path('runs/UKA/MedVLM_2025_03_16_155236')

path_out = Path.cwd()/'results'/path_run_dir.name
path_out.mkdir(parents=True, exist_ok=True)

fontdict = {'fontsize': 11, 'fontweight': 'bold'}

results = []
for label in  CTRATE_Dataset3D.LABELS[:]: # UKA_Dataset3D.LABELS[:]
    print("-------------------", label, "-------------------")
    result = {'Label': label}

    # Read csv with predictions
    df = pd.read_csv(path_run_dir/'results'/f'predictions_{label}.csv')


    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(4,4)) 
    y_pred_lab = np.asarray(df['Prob'])
    y_true_lab = np.asarray(df['GT'])

    tprs, fprs, auc_val, std_auc, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, name=f'{label} AUC:', fontdict=fontdict)
    axis.set_title(f'{label}', fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'roc_{label}.png', dpi=300)
    result['AUC'] = auc_val
    result['AUC_std'] = std_auc

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
    fig.savefig(path_out/f'confusion_matrix_{label}.png', dpi=300)

    results.append(result)
    plt.close()

df = pd.DataFrame(results)
df.to_csv(path_out/'results_summary.csv', index=False)
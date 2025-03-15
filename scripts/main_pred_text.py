import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange
import pandas as pd 
from pathlib import Path 

from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer
from medvlm.utils.gpu import get_gpu_with_max_free_memory
from medvlm.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot, tensor_mask2image


# --------------------- Settings ---------------------
best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
device=torch.device(f'cuda:{best_gpu_index}')


# --------------------- Load the model ---------------------
path_run = Path('runs/UKA/MedVLM_2025_03_07_135708_withCoCo_Vision_LLama/epoch=11-step=5376.ckpt')
model = MedVLM.load_from_checkpoint(path_run)
model.to(device)
# model.eval()
model.save_attn = False


# --------------------- Load the dataset ---------------------
tokenizer = model.tokenizer_y
# ds_test = CTRATE_Dataset3D(split='test', tokenizer=tokenizer)
ds_test = UKA_Dataset3D(split='test', tokenizer=tokenizer)


# --------------------- Setup Dataloader ---------------------
batch_size=1
dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=1)
dl = dm.test_dataloader()


# -------------------- Predict report ---------------------
results = []
counter = 0
for batch in tqdm(dl, total=len(ds_test)//batch_size):
    if counter>100:
        break 
    counter += 1
    img = batch['img'].to(device)
    text = tokenizer.decode(batch['text'])
    uid = batch['uid'] 
    src_key_padding_mask = batch['src_key_padding_mask'].to(device)
    pred_text = model.generate(img=img, src_key_padding_mask=src_key_padding_mask, top_k=1)
    
    
    # -------------- Store results
    for i, label in enumerate(pred_text):
        print("True Report", text[i])
        print("Pred Report", pred_text[i])
        print("------------------")
    
        results.append({
            'UID': uid[i],
            "GT":  text[i],
            "Pred": pred_text[i],
        })
    

results_df = pd.DataFrame(results)
path_out = path_run.parent/'results'
path_out.mkdir(exist_ok=True)
results_df.to_csv(path_out/f"predictions_reports.csv", index=False)
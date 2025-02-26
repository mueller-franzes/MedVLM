import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange

from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer
from medvlm.utils.gpu import get_gpu_with_max_free_memory
from medvlm.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot, tensor_mask2image


# --------------------- Settings ---------------------
best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
print(best_gpu_index, max_free_memory)
device=torch.device(f'cuda:{best_gpu_index}')

# --------------------- Create tokenizer ---------------------
tokenizer = Tokenizer()

# --------------------- Load the dataset ---------------------
# ds_test = CTRATE_Dataset3D(split='test', tokenizer=tokenizer)
ds_test = UKA_Dataset3D(split='test', tokenizer=tokenizer)


# --------------------- Setup Dataloader ---------------------
batch_size=1
dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=1)
dl = dm.test_dataloader()

# --------------------- Load the model ---------------------
model = MedVLM.load_from_checkpoint('runs/UKA/MedVLM_2025_02_25_170749_with_text/epoch=7-step=3632.ckpt')
model.to(device)
# model.eval()
model.save_attn = False


# -------------------- Predict report ---------------------
results = []
for batch in tqdm(dl, total=len(ds_test)//batch_size):
    img = batch['img'].to(device)
    text = tokenizer.decode(batch['text'])
    uid = batch['uid'] 
    src_key_padding_mask = batch['src_key_padding_mask'].to(device)
    pred_text = model.generate(img, x_pad_mask=src_key_padding_mask, y=None, top_k=1)
    
    
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
    
    # --------------- Get attention maps 
    attention_map = model.get_attention_maps()
    source = img # [B, C, D, H, W]
    b, c, *spatial_shape = source.shape
    h = spatial_shape[1]//14
    att_map = rearrange(attention_map, 'b (c d) (h w) -> b c d h w', c=c, h=h) 
    att_map = F.interpolate(att_map, size=spatial_shape, mode='trilinear') # trilinear, area
    save_image(tensor_cam2image(minmax_norm(source[:, :1]), minmax_norm(att_map[:, :1]), alpha=0.5), f"overlay.png", normalize=False)

import torch
from transformers import AutoTokenizer 
import numpy as np
from tqdm import tqdm

from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer

tokenizer = Tokenizer()

# ds_train = CTRATE_Dataset3D(split='train', tokenizer=tokenizer,
#                             random_flip=True, random_noise=True, random_center=True, random_rotate=True)
ds_train = UKA_Dataset3D(tokenizer=tokenizer)
dm = DataModule(ds_train=ds_train, batch_size=2, num_workers=0, num_train_samples=10)
dl = dm.train_dataloader() 

device=torch.device('cuda')

model = MedVLM(tokenizer_y=tokenizer)
# model = MedVLM.load_from_checkpoint('runs/UKA/MedVLM_2025_01_10_203931/epoch=17-step=66366.ckpt')
model.to(device)


for idx, batch in tqdm(enumerate(iter(dl))):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)

# for n in range(10):
#     item = ds_train[n]
#     pred_text = model.generate(item['img'][None].to(device), x_pad_mask=item['src_key_padding_mask'][None].to(device), y=None, top_k=1)
#     for b in pred_text:
#         print("True Report", tokenizer.decode(item['text'][None])[0])
#         print("Pred Report", b)
#         print("------------------")
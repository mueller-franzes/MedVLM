import torch
from transformers import AutoTokenizer 
import numpy as np

from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer

tokenizer = Tokenizer()

ds_train = CTRATE_Dataset3D(split='train', tokenizer=tokenizer,
                            random_flip=True, random_noise=True, random_center=True, random_rotate=True)
dm = DataModule(ds_train=ds_train, batch_size=2, num_workers=0)
dl = dm.train_dataloader() 

device=torch.device('cuda')

model = MedVLM(tokenizer_y=tokenizer)
# model = MedVLM.load_from_checkpoint('runs/CTRATE/MedVLM_2025_01_09_233725/epoch=209-step=210000.ckpt')
model.to(device)


for idx, batch in enumerate(iter(dl)):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)


# pred_text = model.generate(ds_train[0]['img'][None].to(device), y=None, top_k=100)
# for b in pred_text:
#     print(b)
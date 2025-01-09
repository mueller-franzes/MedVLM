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

device=torch.device('cuda:3')

model = MedVLM(tokenizer_y=tokenizer)
model.to(device)


# for idx, batch in enumerate(iter(dl)):
#     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#     loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)


model.generate(ds_train[0]['img'][None].to(device), y=None, max_new_tokens=100)
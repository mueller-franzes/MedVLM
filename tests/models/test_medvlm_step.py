import torch
import numpy as np
from tqdm import tqdm


from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer


tokenizer = Tokenizer(model_name="meta-llama/Llama-3.2-1B") # model_name="meta-llama/Llama-3.2-1B"

# ds_test = CTRATE_Dataset3D(split='test', tokenizer=tokenizer)
ds_test = UKA_Dataset3D(split='test', tokenizer=tokenizer)
# ds_test.LABEL = "Pleural effusion"
ds_test.LABEL = "Karzinom"

device=torch.device('cuda:8')

model = MedVLM(tokenizer_y=tokenizer, use_llm=True)
# model = MedVLM.load_from_checkpoint('runs/UKA/MedVLM_2025_02_23_215831/epoch=12-step=5902.ckpt')
model.to(device)


dm = DataModule(ds_test=ds_test, batch_size=1, num_workers=0, num_train_samples=10)
dl = dm.test_dataloader() 
for idx, batch in tqdm(enumerate(iter(dl))):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)


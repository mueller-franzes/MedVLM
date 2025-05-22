import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #solve out oif memory on login node

import torch
import numpy as np
from tqdm import tqdm
import gc

from tests.helper  import print_mem
from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer
from medvlm.utils.gpu import get_gpu_with_max_free_memory

def get_dataset(name):
    if name == 'CTRATE':
        return CTRATE_Dataset3D
    elif name == 'UKA':
        return UKA_Dataset3D
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
# tokenizer = Tokenizer(model_name="meta-llama/Llama-3.2-1B") #
# tokenizer = Tokenizer(model_name="GerMedBERT/medbert-512") #German
# tokenizer = Tokenizer(model_name="emilyalsentzer/Bio_ClinicalBERT")
tokenizer = Tokenizer(model_name='microsoft/BiomedVLP-CXR-BERT-specialized')
#other options: bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16, dmis-lab/biobert-v1.1

#CT Clip uses: tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
# text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

state = "val"
ds_train = get_dataset("CTRATE")(split='train', random_flip=True, random_noise=True, random_center=True, random_rotate=True, tokenizer=tokenizer)
ds_val = get_dataset("CTRATE")(split='val', tokenizer=tokenizer)
ds_test = get_dataset("CTRATE")(split='test', tokenizer=tokenizer)

ds = CTRATE_Dataset3D(split=state, tokenizer=tokenizer)
# ds_test = UKA_Dataset3D(split='test', tokenizer=tokenizer)
ds.LABEL = "Pleural effusion"
# ds_test.LABEL = "Karzinom"

best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
device=torch.device(f'cuda:{best_gpu_index}')

model = MedVLM(tokenizer_y=tokenizer, data_type="CT", use_llm=False, text_encoder="microsoft/BiomedVLP-CXR-BERT-specialized", only_cl=True #, backbone_type="resnet", model_size=34
               )
# model = MedVLM.load_from_checkpoint('runs/UKA/MedVLM_2025_02_23_215831/epoch=12-step=5902.ckpt')

# Set model to train mode
model.train()

# Get optimizer
optimizer = model.configure_optimizers()
if isinstance(optimizer, (tuple, list)):
    optimizer = optimizer[0]  # unpack if needed

model.to(device)


dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_val, batch_size=2, num_workers=0, num_train_samples=10)
if state == "train":
    dl = dm.train_dataloader()
elif state == "val":
    dl = dm.val_dataloader()
elif state == "test":
    dl = dm.test_dataloader()
else:
    raise Exception("Invalid state.")

for idx, batch in tqdm(enumerate(iter(dl))):
    print_mem(f"  Before _step, idx={idx}")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    # --- LOSS BEFORE BACKWARD ---
    with torch.no_grad():
        loss_before = model._step(batch, batch_idx=idx, state=state, step=idx*dm.batch_size)
        loss_before_val = loss_before.item()
    print("Loss before backward:", loss_before_val)
    print_mem(f"  After _step, idx={idx}")

    # --- BACKWARD + STEP ---
    optimizer[0].zero_grad()
    loss = model._step(batch, batch_idx=idx, state=state, step=idx * dm.batch_size)
    loss.backward()
    optimizer[0].step()

    # --- LOSS AFTER UPDATE ---
    with torch.no_grad():
        loss_after = model._step(batch, batch_idx=idx, state=state, step=idx * dm.batch_size)
        loss_after_val = loss_after.item()
    print("Loss after update:", loss_after_val)

    # --- Check if loss decreased ---
    if loss_after_val < loss_before_val:
        print("Loss decreased.")
    else:
        print("Loss did NOT decrease. Could be okay, but watch for trends.")

    # del batch
    # gc.collect()
    # torch.cuda.empty_cache()
    # print_mem(f"  After del, idx={idx}")



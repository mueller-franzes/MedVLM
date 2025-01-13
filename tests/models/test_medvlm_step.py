import torch
from transformers import AutoTokenizer 
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule 
from medvlm.models.tokenizer import Tokenizer

tokenizer = Tokenizer()

ds_test = CTRATE_Dataset3D(split='test', tokenizer=tokenizer)
# ds_test = UKA_Dataset3D(split='test', tokenizer=tokenizer)
ds_test.LABEL = "Pleural effusion"


device=torch.device('cuda')

# model = MedVLM(tokenizer_y=tokenizer)
model = MedVLM.load_from_checkpoint('runs/CTRATE/MedVLM_2025_01_12_130944/epoch=94-step=95000.ckpt')
model.to(device)
model.eval()

# dm = DataModule(ds_test=ds_test, batch_size=2, num_workers=0, num_train_samples=10)
# dl = dm.test_dataloader() 
# for idx, batch in tqdm(enumerate(iter(dl))):
#     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#     loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)

# for n in range(10):
#     item = ds_test[n]
#     pred_text = model.generate(item['img'][None].to(device), x_pad_mask=item['src_key_padding_mask'][None].to(device), y=None, top_k=10)
#     for b in pred_text:
#         print("True Report", tokenizer.decode(item['text'][None])[0])
#         print("Pred Report", b)
#         print("------------------")

for n in range(10, 30):

    item = ds_test[n]
    img = item['img'][None].repeat(2, 1, 1, 1, 1).to(device)
    src_key_padding_mask = item['src_key_padding_mask'][None].repeat(2, 1).to(device)
    text1 = model.tokenizer_y(f"No {ds_test.LABEL}.")
    text2 = model.tokenizer_y(f"{ds_test.LABEL}.")
    text = torch.stack([text1, text2])[:, :-1].to(device)
    with torch.no_grad():
        _ = model.forward(img=img, text=text, src_key_padding_mask=src_key_padding_mask)

    # memory_cls = F.normalize(model.memory_cls, dim=1)  # Normalize for cosine similarity
    # tgt_cls = F.normalize(model.tgt_cls, dim=1)     # Normalize for cosine similarity
    # logits = torch.mm(memory_cls, tgt_cls.t())  # Shape: [B, B]

    cs = F.cosine_similarity(model.memory_cls, model.tgt_cls, dim=1)
    print(f"True {item['uid']} Report", tokenizer.decode(item['text'][None])[0])
    print("Pred Prob", cs)
    print("Pred Label ", cs.argmax().cpu().item())
    print("True Label", item["label"])
    print("-----------")
import torch
from transformers import AutoTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path 
import argparse
import torch.nn.functional as F
from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule
from medvlm.models.tokenizer import Tokenizer
from medvlm.utils.gpu import get_gpu_with_max_free_memory
from torchvision.utils import save_image
from medvlm.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot, tensor_mask2image

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR
from medvlm.models.tokenizer import Tokenizer

def get_dataset(name):
    if name == 'CTRATE':
        return CTRATE_Dataset3D
    elif name == 'UKA':
        return UKA_Dataset3D
    else:
        raise ValueError(f"Unknown dataset: {name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="UKA")
    parser.add_argument('--model', type=str, default="MedVLM")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--path_run', type=str, default="runs/UKA/MedVLM_2025_04_01_183114_Llama/epoch=94-step=42560.ckpt")
    parser.add_argument('--dataset_type', type=str, default='test')
    args = parser.parse_args()

    path_run = Path(args.path_run)
    path_out = path_run.parent/'results_report'
    # path_out_weight.mkdir(parents=True, exist_ok=True)
    best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
    device=torch.device(f'cuda:{best_gpu_index}')

    # Load the model
    if args.dataset == "UKA":
        tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512")
    elif args.dataset == "CTRATE":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
        CXRBertTokenizer = type(tokenizer)
        torch.serialization.add_safe_globals([CXRBertTokenizer, AdamW, LinearLR, Tokenizer])

    # ckpt = torch.load(args.path_run, weights_only=False)
    model = MedVLM.load_from_checkpoint(path_run, strict=False)
    model.to(device, non_blocking=True)
    model.eval()

    # Load the dataset
    tokenizer = model.tokenizer_y
    ds_test = get_dataset(args.dataset)(split=args.dataset_type, tokenizer=tokenizer)
    # ds_test.item_pointers = ds_test.item_pointers[:10]

    # Set up the dataloader
    batch_size=args.batch_size
    dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=16)
    dl = dm.test_dataloader(shuffle=True)

    # Prepare storage for results
    results = []
    i2t_results = []
    t2i_results = []

    # Iterate over the dataloader
    for batch in tqdm(dl, total=len(ds_test)//batch_size):
        imgs = batch['img'].to(device, non_blocking=True)
        labels = batch['label']
        text = batch['text'].to(device)
        # # Prepare text prompts
        # if args.dataset == "UKA":
        #     text1 = tokenizer(f"Kein {ds_test.LABEL}") # No or Kein [512,]
        #     text2 = tokenizer(f"{ds_test.LABEL}")
        # else:
        #     text1 = tokenizer(f"{ds_test.LABEL} is not present")
        #     text2 = tokenizer(f"{ds_test.LABEL} is present")


        # text = torch.stack([text1, text2])[:, :-1].to(device) # Remove the last token (eos)

        with torch.no_grad():
            probs_i2t, probs_t2i = model.compute_similiarty(text, imgs)
            
            for i, uid in enumerate(batch['uid']):
                pred = probs_i2t[i].argmax().cpu().item() #maximum for each row i, find best text for fixed image
                gt = np.zeros(args.batch_size, dtype=int)
                np.put(gt,i,1)
                i2t_results.append({ #choose best text for fized image
                    'UID': uid,
                    "GT": gt,
                    "GT_index": i,
                    "Pred": pred,
                    "Prob": probs_i2t[i].cpu().numpy(),
                })
                pred = probs_t2i[:,i].argmax().cpu().item() #Softmax calculated along column, find max in column, find best image for fixed text
                t2i_results.append({ #choose best image for fixed text
                    'UID': uid,
                    "GT": gt,
                    "GT_index": i,
                    "Pred": pred,
                    "Prob": probs_t2i[:,i].cpu().numpy(),
                })

            
    # Save results to a CSV file
    results_i2t = pd.DataFrame(i2t_results)
    results_t2i = pd.DataFrame(t2i_results)

    results_i2t.to_csv(path_out/f"predictions_i2t_{args.dataset_type}.csv", index=False)
    results_t2i.to_csv(path_out/f"predictions_t2i_{args.dataset_type}.csv", index=False)

    print(f"Predictions saved to {path_out}/predictions_i2t_{args.dataset_type}.csv")

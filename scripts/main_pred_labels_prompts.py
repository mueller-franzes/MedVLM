import torch
from transformers import AutoTokenizer, BertModel
import pandas as pd
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--path_run', type=str, default="runs/UKA/MedVLM_2025_04_01_183114_Llama/epoch=94-step=42560.ckpt")
    args = parser.parse_args()

    path_run = Path(args.path_run)
    path_out = path_run.parent/'results'
    path_out_weight = path_out/'weights'
    path_out_weight.mkdir(parents=True, exist_ok=True)
    best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
    device=torch.device(f'cuda:{best_gpu_index}')

    # Load the model
    if args.dataset == "UKA":
        raise(NotImplementedError)
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
    ds_test = get_dataset(args.dataset)(split='test', tokenizer=tokenizer)
    # ds_test = get_dataset(args.dataset)(split='train', tokenizer=tokenizer)
    # ds_test.item_pointers = ds_test.item_pointers[:1000]


    # Iterate over all labels
    for label_name in  ds_test.LABELS[:]: # ds_test.LABELS[:]
        ds_test.LABEL = label_name
        #Text Prompts:
        texts = [[f"{ds_test.LABEL} is not present", f"{ds_test.LABEL} is present"],
                 [f"{ds_test.LABEL} was not observed", f"{ds_test.LABEL} was observed"],
                 [f"{ds_test.LABEL} was not detected", f"{ds_test.LABEL} was detected"],
                 [f"No significant {ds_test.LABEL} detected", f"Significant {ds_test.LABEL} detected"],
                 [f"No significant {ds_test.LABEL} detected", f"{ds_test.LABEL} was observed"]
                 ]
        for j, prompt in enumerate(texts):
            # Set up the dataloader
            batch_size=args.batch_size
            dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=16)
            dl = dm.test_dataloader()

            # Prepare storage for results
            results = []

            # Iterate over the dataloader
            for batch in tqdm(dl, total=len(ds_test)//batch_size):
                imgs = batch['img'].to(device, non_blocking=True)
                labels = batch['label']

                # Prepare text prompts
                text1 = tokenizer(prompt[0])
                text2 = tokenizer(prompt[1])
                text = torch.stack([text1, text2])[:, :-1].to(device) # Remove the last token (eos)

                with torch.no_grad():
                    probs, probs_t2i = model.compute_similiarty(text, imgs)

                    # Store results
                    for i, label in enumerate(labels):
                        results.append({
                            'UID': batch['uid'][i],
                            "GT": label.item(),
                            "Pred": probs[i].argmax().cpu().item(),
                            "Prob": probs[i, 1].cpu().item(),
                        })

            # Save results to a CSV file
            results_df = pd.DataFrame(results)
            results_df.to_csv(path_out/f"predictions_{ds_test.LABEL}_{j}.csv", index=False)

            print("Predictions saved to predictions.csv")

import torch
from transformers import AutoTokenizer
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
    model = MedVLM.load_from_checkpoint(path_run, strict=False)
    model.to(device)
    model.eval()

    # Load the dataset
    tokenizer = model.tokenizer_y
    ds_test = get_dataset(args.dataset)(split='test', tokenizer=tokenizer)


    # Iterate over all labels
    for label_name in  ds_test.LABELS[:]: # ds_test.LABELS[:]
        ds_test.LABEL = label_name


        # Set up the dataloader
        batch_size=args.batch_size
        dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=16)
        dl = dm.test_dataloader()

        # Prepare storage for results
        results = []

        # Iterate over the dataloader
        for batch in tqdm(dl, total=len(ds_test)//batch_size):
            imgs = batch['img'].to(device)
            labels = batch['label']

            # Prepare text prompts
            text1 = tokenizer(f"Kein {ds_test.LABEL}") # No or Kein [512,]
            text2 = tokenizer(f"{ds_test.LABEL}")

            text = torch.stack([text1, text2])[:, :-1].to(device) # Remove the last token (eos)

            with torch.no_grad():
                probs = model.compute_similiarty(text, imgs)

                # probs, weight_slice = model.compute_similiarty_attention(text, imgs)
                # weight_slice = weight_slice.unsqueeze(-1).unsqueeze(-1).expand(imgs.shape) # [B, C, D] -> [B, C, D, H, W]
                # save_image(tensor_cam2image(minmax_norm(imgs[:, 1:2].cpu()), minmax_norm(weight_slice[:, 1:2].cpu()), alpha=0.5), 
                #         path_out_weight/f"overlay_{batch['uid'][0]}_{label_name}_slice.png", normalize=False)

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
        results_df.to_csv(path_out/f"predictions_{ds_test.LABEL}.csv", index=False)

        print("Predictions saved to predictions.csv")

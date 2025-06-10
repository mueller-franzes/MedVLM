import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer 
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
    parser.add_argument('--path_run', type=str, default="/home/ve001107/MedVLM/runs/CTRATE/MedVLM_2025_05_25_150747_trainable/best-epoch=20-val/contrastive_real=3.10.ckpt")
    parser.add_argument('--prompt', type=str, default='A')
    args = parser.parse_args()

    assert(args.dataset == "UKA" or args.dataset == "CTRATE")

    path_run = Path(args.path_run)
    path_out = path_run.parent/f"results_prompt{args.prompt}"
    path_out_weight = path_out/'weights'
    path_out_weight.mkdir(parents=True, exist_ok=True)
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
    ds_test = get_dataset(args.dataset)(split='test', tokenizer=tokenizer)
    # ds_test = get_dataset(args.dataset)(split='train', tokenizer=tokenizer)
    # ds_test.item_pointers = ds_test.item_pointers[:1000]

    #use prompts from file
    if args.prompt == 'C':
        prompts = pd.read_excel("/home/ve001107/MedVLM/MedVLM/prompts/CTRATE/Chatgptprompts.xlsx", index_col='Prompt')
        # prompts.set_axis(['index', 'positive', 'negative'], axis='columns')
        # prompts.set_index('Prompt')

    # Iterate over all labels
    for label_name in  ds_test.LABELS[:]: # ds_test.LABELS[:]
        ds_test.LABEL = label_name


        # Set up the dataloader
        batch_size=args.batch_size
        dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=16)
        dl = dm.test_dataloader()

        # Prepare text prompts
        if args.dataset == "UKA":
            text1 = tokenizer(f"Kein {ds_test.LABEL}") # No or Kein [512,]
            text2 = tokenizer(f"{ds_test.LABEL}")
        else:
            match args.prompt:
                case 'A':
                    text1 = tokenizer(f"{ds_test.LABEL} is not present")
                    text2 = tokenizer(f"{ds_test.LABEL} is present")
                case 'B':
                    text1 = tokenizer(f"No significant {ds_test.LABEL} detected")
                    text2 = tokenizer(f"Significant {ds_test.LABEL} detected")
                case 'C':
                    text1 = tokenizer(prompts.loc[ds_test.LABEL, "Negative prompt"])
                    text2 = tokenizer(prompts.loc[ds_test.LABEL, "Positive prompt"])
                case 'D':
                    texts = [[f"{ds_test.LABEL} is not present", f"{ds_test.LABEL} is present"],
                            [f"{ds_test.LABEL} was not observed", f"{ds_test.LABEL} was observed"],
                            [f"{ds_test.LABEL} was not detected", f"{ds_test.LABEL} was detected"],
                            [f"No significant {ds_test.LABEL} detected", f"Significant {ds_test.LABEL} detected"],
                            [f"No significant {ds_test.LABEL} detected", f"{ds_test.LABEL} was observed"]
                            ]
                    pos_tokenized = []
                    neg_tokenized = []
                    for [negative, positive] in texts:
                        pos_tokenized.append(tokenizer(positive))
                        neg_tokenized.append(tokenizer(negative))
                    negative = torch.stack(neg_tokenized)[:, :-1].to(device)
                    positive = torch.stack(pos_tokenized)[:, :-1].to(device)
                    with torch.no_grad():
                        neg_emb, _ = model.forward_text(negative)
                        pos_emb, _ = model.forward_text(positive)
                    neg_emb = neg_emb.mean(dim=0)
                    pos_emb = pos_emb.mean(dim=0)
                    text_cls = torch.stack([neg_emb, pos_emb])
                case _:
                    raise(NotImplementedError, f"Prompt {args.prompt} not defined")
        
        if not args.prompt == 'D':
            text = torch.stack([text1, text2])[:, :-1].to(device) # Remove the last token (eos)

        # Prepare storage for results
        results = []

        # Iterate over the dataloader
        for batch in tqdm(dl, total=len(ds_test)//batch_size):
            imgs = batch['img'].to(device, non_blocking=True)
            labels = batch['label']            

            with torch.no_grad():
                if args.prompt == 'D':
                    img_cls, _ = model.forward_vision(imgs)
                    probs, probs_t2i = model.compute_similarity_embed(text_cls=text_cls, vision_cls=img_cls)
                else:
                    probs, probs_t2i = model.compute_similiarty(text, imgs)

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
        results_path = path_out/f"predictions_{ds_test.LABEL}.csv"
        results_df.to_csv(results_path, index=False)

        print(f"Predictions saved to {results_path}")

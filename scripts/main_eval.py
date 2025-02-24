import torch
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from pathlib import Path 
import torch.nn.functional as F
from medvlm.models.medvlm import MedVLM
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datamodule import DataModule
from medvlm.models.tokenizer import Tokenizer
from medvlm.utils.gpu import get_gpu_with_max_free_memory

tokenizer = Tokenizer()

ds_test = CTRATE_Dataset3D(split='test', tokenizer=tokenizer)
for label in ds_test.LABELS[7:8]:
    ds_test.LABEL = label

    best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
    print(best_gpu_index, max_free_memory)
    device=torch.device(f'cuda:{best_gpu_index}')

    # Load the model
    path_run = Path('runs/CTRATE/MedVLM_2025_01_24_140928_w_contrastive/epoch=5-step=3624.ckpt')
    model = MedVLM.load_from_checkpoint(path_run)
    model.to(device)
    model.eval()

    # Set up the dataloader
    batch_size=4
    dm = DataModule(ds_test=ds_test, batch_size=batch_size, num_workers=16)
    dl = dm.test_dataloader()

    # Prepare storage for results
    results = []

    # Iterate over the dataloader
    for batch in tqdm(dl, total=len(ds_test)//batch_size):
        imgs = batch['img'].to(device)
        src_key_padding_masks = batch['src_key_padding_mask'].to(device)
        labels = batch['label']

        # Repeat imgs and src_key_padding_masks to align with text prompts
        imgs = imgs.repeat_interleave(2, dim=0)
        src_key_padding_masks = src_key_padding_masks.repeat_interleave(2, dim=0)

        # Prepare text prompts
        text1 = model.tokenizer_y(f"No {ds_test.LABEL}")
        text2 = model.tokenizer_y(f"{ds_test.LABEL}")
        text = torch.stack([text1, text2])[:, :-1].to(device)

        # Repeat text for the batch
        text = text.repeat(len(batch['img']), 1)

        with torch.no_grad():
            # Forward pass for repeated images and text
            _ = model.forward(img=imgs, text=text, src_key_padding_mask=src_key_padding_masks)
            image_features, text_features = model.memory_cls, model.tgt_cls
            
            # image_features = model.linear_a(model.encode_img(imgs, src_key_padding_masks)[:, -1]).repeat_interleave(2, dim=0)
            # text_features = model.linear_b(model.encode_text(text)[:, -1])

            # Extract features and normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            temperature = model.cliploss.logit_scale.exp()

            # Calculate logits and probabilities for each image against both text prompts
            logits = torch.sum(image_features * text_features, dim=-1).view(-1, 2) * temperature
            probs = logits.softmax(dim=-1)

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
    path_out = path_run.parent/'results'
    path_out.mkdir(exist_ok=True)
    results_df.to_csv(path_out/f"predictions_{ds_test.LABEL}.csv", index=False)

    print("Predictions saved to predictions.csv")

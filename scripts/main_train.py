import argparse
from pathlib import Path
import os 
os.environ["WANDB__SERVICE_WAIT"] = "300"
from datetime import datetime
import wandb 
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from medvlm.models.tokenizer import Tokenizer
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.data.datamodule import DataModule
from medvlm.models.medvlm import MedVLM

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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tokenizer', type=str, default="GerMedBERT/medbert-512") # "GerMedBERT/medbert-512" "meta-llama/Llama-3.2-1B", google/gemma-3-1b-pt
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # path_run_root = Path('/hpcwork/p0020933/workspace_gustav/models/medvlm')
    # path_run_root = Path('/work/gm511073/workspace_gustav/models/medvlm')
    path_run_root = Path.cwd()
    run_name =  f'{args.model}_{current_time}_trainable'
    path_run_dir = path_run_root / 'runs' / args.dataset / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' # if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('medium')

    # ------------ Load Data ----------------
    tokenizer = Tokenizer(model_name=args.tokenizer)
    ds_train = get_dataset(args.dataset)(split='train', random_flip=True, random_noise=True, random_center=True, random_rotate=True, tokenizer=tokenizer)
    ds_val = get_dataset(args.dataset)(split='val', tokenizer=tokenizer)
    ds_test = get_dataset(args.dataset)(split='test', tokenizer=tokenizer)

    samples = len(ds_train) + len(ds_val)
    batch_size = args.batch_size #1 if args.dataset == 'CTRATE' else 2
    accumulate_grad_batches = 1 #2 if args.dataset == 'CTRATE' else 1
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    # class_counts = ds_train.df[ds_train.LABEL].value_counts()
    # class_weights = 0.5 / class_counts
    # weights = ds_train.df[ds_train.LABEL].map(lambda x: class_weights[x]).values

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        # weights=weights,
        num_workers=16,#args.batch_size,
        # num_train_samples=min(len(ds_train), 2000) # WARNING: Ignored for DDP 
    )

    # ------------ Initialize Model ------------
    model = MedVLM(tokenizer_y=tokenizer)
    # model = MedVLM.load_from_checkpoint('runs/UKA/MedVLM_2025_03_16_155236/epoch=5-step=2688.ckpt', strict=False)

    
    # -------------- Training Initialization ---------------
    to_monitor = "val/contastive_real"
    min_max = "min"
    log_every_n_steps = 50
    logger = WandbLogger(project=f'MedVLM', group=args.dataset, name=run_name, log_model=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=50,
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        # gradient_clip_val =0.5,
        # replace_sampler_ddp=True, # WARNING: For DDP: Random DDP Sample is used unless set True here 
        strategy=DDPStrategy(static_graph=True), # static_graph=True required if gradient checkpoint is used  
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor], # early_stopping
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=min(len(ds_val), 200),
        # limit_val_batches = 0,
        limit_train_batches =  0.5 if args.dataset=="CTRATE" else None,
        # max_epochs=1,
        num_sanity_val_steps=2,
        logger=logger
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    # model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)

    wandb.finish(quiet=True)
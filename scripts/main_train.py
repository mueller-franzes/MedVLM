import argparse
from pathlib import Path
from datetime import datetime
import wandb 
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from medvlm.models.tokenizer import Tokenizer
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datamodule import DataModule
from medvlm.models.medvlm import MedVLM




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="CTRATE")
    parser.add_argument('--model', type=str, default="MedVLM")
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # path_run_root = Path('/hpcwork/p0020933/workspace_gustav/models/medvlm')
    # path_run_root = Path('/work/gm511073/workspace_gustav/models/medvlm')
    path_run_root = Path.cwd()
    path_run_dir = path_run_root / 'runs' / args.dataset / f'{args.model}_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' # if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Load Data ----------------
    tokenizer = Tokenizer()
    ds_train = CTRATE_Dataset3D(split='train', random_flip=True, random_noise=True, random_center=True, random_rotate=True, tokenizer=tokenizer)
    ds_val = CTRATE_Dataset3D(split='val', tokenizer=tokenizer)
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 1 #1 if args.dataset == 'CTRATE' else 2
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
        num_workers=16,
        num_train_samples=min(len(ds_train), 2000)
    )

    # ------------ Initialize Model ------------
    model = MedVLM(tokenizer_y=tokenizer)

    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(project=f'MedVLM', name=f'{args.model}_{current_time}', log_model=False)
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
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=min(len(ds_val), 200),
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    # model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)

    wandb.finish(quiet=True)
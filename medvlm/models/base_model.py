from pathlib import Path
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import  torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import numpy as np


from .utils.losses import CLIPLoss


class VeryBasicModel(pl.LightningModule):
    def __init__(self, save_hyperparameters=True):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1


    def forward(self, x, cond=None):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError
    
    def _epoch_end(self, state:str):
        return 

    def training_step(self, batch: dict, batch_idx: int ):
        self._step_train += 1 
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val )

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None: 
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self, outputs) -> None:
        self._epoch_end("test")

    def compute_grad_norm(self, model):
        # total_norm_sq = 0.0
        # for param in module.parameters():
        #     if param.grad is not None:
        #         total_norm_sq += param.grad.norm(2).item() ** 2
        # return total_norm_sq ** 0.5
        norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms.append(param.grad.norm().item())
        norms = np.array(norms)
        norm = np.mean(norms) if norms.any() else 0
        return norm

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)  

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
     
        return self.load_weights(checkpoint["state_dict"], **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 




class BasicModel(VeryBasicModel):
    def __init__(
        self, 
        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-3, 'weight_decay':1e-2},
        lr_scheduler= None,
        lr_scheduler_kwargs={},
        save_hyperparameters=True
    ):
        super().__init__(save_hyperparameters=save_hyperparameters)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            lr_scheduler_config  = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [lr_scheduler_config ]
        else:
            return [optimizer]
        



class BasicVLM(BasicModel):
    def __init__(
        self, 
        tokenizer_y,
        optimizer = torch.optim.AdamW,
        optimizer_kwargs ={'lr':5e-4},
        lr_scheduler= lr_scheduler.LinearLR, 
        # lr_scheduler= lr_scheduler.CosineAnnealingWarmRestarts, 
        lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':1000},
        # lr_scheduler_kwargs={'T_0':5000, 'T_mult':2},
        only_cl = False, # only contrastive loss
        save_hyperparameters=True
    ):
        super().__init__(
            optimizer, 
            optimizer_kwargs, 
            lr_scheduler, 
            lr_scheduler_kwargs, 
            save_hyperparameters=save_hyperparameters
        )

        self.tokenizer_y = tokenizer_y
        self.cliploss = CLIPLoss()
        self.only_cl = only_cl 

        # self.ce = nn.ModuleDict({state:CrossEntropy() for state in ["train_", "val_", "test_"]}) # 'train' not allowed as key
        # self.acc = nn.ModuleDict({state:Accuracy(task='multiclass', num_classes=3) for state in ["train_", "val_", "test_"]})
    
    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        x, y = batch['img'], batch['text']
    
        if dist.is_initialized():
            self.batch_size = x.shape[0] * dist.get_world_size()
        else:
            self.batch_size = x.shape[0]  # Fallback for non-distributed mode
 
        logging_dict = {}


        # ------------------- Run Model --------------------------
        memory_cls, text_cls, text_logits, vision_logits  = self.forward(img=x, text=y[:, :-1]) # [B, T, C]


        # ------------------------- Compute Loss ---------------------------  
        loss = 0      
        # -------- Compute Contrastive Loss 
        image_embeddings = memory_cls
        text_embeddings = text_cls
        #TODO: check if cls can be deleted to free memory
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Gather embeddings from all GPUs
        if dist.is_initialized():
            image_embeddings = self.all_gather(image_embeddings, sync_grads=True) # [World_size, B, C]
            text_embeddings = self.all_gather(text_embeddings, sync_grads=True)
            # Ensure all gathered features are correctly reshaped
            image_embeddings = image_embeddings.view(-1, image_embeddings.shape[-1]) # [World_size, B, C] -> [B*World_size, C]
            text_embeddings = text_embeddings.view(-1, text_embeddings.shape[-1])

        contrastive_loss, _, _ = self.cliploss(image_embeddings, text_embeddings)
        loss += contrastive_loss

        # Compute contrastive loss without temperature scaling
        with torch.no_grad():
            contrastive_loss_fix, texts_loss, images_loss  = self.cliploss(image_embeddings, text_embeddings, temperature=1)

        # -------- Compute Cross Entropy Loss for text generation
        if not self.only_cl:
            # ce_text = 0
            y = y[:, 1:] # Remove <SOS> 
            y_pred = text_logits.transpose(1, 2) # [B, N, C] ->   [B, C, N]
            ce_text = F.cross_entropy(y_pred, y, ignore_index=self.tokenizer_y.pad_token_id) 
            loss += ce_text

            # ------- Compute Cross Entropy Loss for image generation 
            vision_target = self.vision_emb.detach() # Ugly hack to get the target
            log_pred = F.log_softmax(vision_logits, dim=-1)
            vision_target = F.softmax(vision_target, dim=-1)
            ce_vision = -torch.sum(log_pred * vision_target, dim=-1)
            ce_vision[self._vision_padding_mask] = 0
            ce_vision = ce_vision.mean()
            loss += ce_vision
        
 

        # ------------------- Log Scalars ----------------------
        logging_dict['loss'] = loss
        logging_dict['contrastive'] = contrastive_loss
        logging_dict['contrastive_real'] = contrastive_loss_fix
        logging_dict['image2text'] = images_loss
        logging_dict['text2image'] = texts_loss
        logging_dict['temperature'] = self.cliploss.logit_scale.exp()
        if not self.only_cl:
            logging_dict['ce_text'] = ce_text
            logging_dict['ce_vision'] = ce_vision
            
        # --------------------- Log Gradients ----------------------------------
        if state=="train":
            grad_norms = {}            
            grad_norms['cls_emb'] = self.cls_emb.grad.norm().item() if (self.cls_emb.grad is not None) else 0
            grad_norms['cls_logits'] = self.cls_logits.weight.grad.norm().item() if self.cls_logits.weight.grad is not None else 0
            grad_norms['cliploss_logit_scale'] = self.cliploss.logit_scale.grad.norm().item() if self.cliploss.logit_scale.grad is not None else 0
            grad_norms['vision_pos_emb'] = self.vision_pos_emb.weight.grad.norm().item() if self.vision_pos_emb.weight.grad is not None else 0
            grad_norms['text_vision_proj'] = self.text_vision_proj.weight.grad.norm().item() if self.text_vision_proj.weight.grad is not None else 0
            grad_norms['vision'] = self.compute_grad_norm(self.vision_encoder.backbone)
            grad_norms['multi_encoder'] = self.compute_grad_norm(self.multi_encoder)
            for metric_name, metric_val in grad_norms.items():
                self.log(f"grad_norm/{metric_name}", metric_val.detach() if hasattr(metric_val, "detach") else metric_val,
                         batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=metric_name=="loss", sync_dist=True)               
             

        # --------------------- Compute Metrics  -------------------------------
        # pred_tokens = self.logits2tokens(logits) 
        with torch.no_grad():
            # num_tokens_y = y.size(1)
            # mask_padding_y = y == self.tokenizer_y.pad_token_id
            # pred_tokens_y = pred_tokens[:, -num_tokens_y:] 
            # logging_dict["acc"] = torch.sum(pred_tokens_y[~mask_padding_y] == y[~mask_padding_y])/torch.numel(y[~mask_padding_y])
  
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val.detach() if hasattr(metric_val, "detach") else metric_val,
                         batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=metric_name=="loss", sync_dist=True)               
            
        return logging_dict['loss'] 



    @torch.no_grad()
    def generate(self, text=None, img=None, src_key_padding_mask = None, return_logits=False, **kwargs):
        pred_tokens, logits = self._generate(text=text, img=img, src_key_padding_mask=src_key_padding_mask, **kwargs)
        pred = self.tokenizer_y.decode(pred_tokens)
        if return_logits:
            return pred, logits
        return pred
    

    @torch.no_grad()
    def _generate(self, text=None, img=None, src_key_padding_mask=None, max_new_tokens=None, batch_size=1, **kwargs):
        max_new_tokens = self.tokenizer_y.max_length if max_new_tokens is None else max_new_tokens 
        bos_token_id = self.tokenizer_y.bos_token_id
        eos_token_id = self.tokenizer_y.eos_token_id

        tokens = text
        if tokens is None:
            tokens = torch.tensor([bos_token_id], device=self.device).repeat(batch_size, 1)

        n=0
        while True:
            _, memory = self.forward_vision(img, src_key_padding_mask=src_key_padding_mask)
            _, text_emb = self.forward_text(tokens) 
            logits = self.forward_vision_text(memory, text_emb)

            logits = logits[:, -1:] # [B, N, C] -> [B, 1, C]
            next_token = self.logits2tokens(logits, **kwargs)
            tokens = torch.cat((tokens, next_token), dim=1)
            logits_seq = torch.cat((logits_seq, logits), dim=1) if n > 0  else logits
            n = n+1
            if (next_token == eos_token_id) or n>=max_new_tokens :
                break 
        
        # tokens = tokens[:, 1:] # Maybe remove SOS or x 
        return tokens, logits_seq 
    
    
    def logits2tokens(self, logits, top_k=1, temperature=1.0):
        # logits: [B, N, C] 
        C = logits.size(-1)

        # scale by desired temperature
        logits = logits/ temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            if isinstance(top_k, float):
                top_k =  int(top_k*C)
            v, _ = torch.topk(logits, min(top_k, C))
            logits[logits < torch.min(v, dim=-1, keepdim=True).values] = -float('Inf')

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution
        token = torch.stack([torch.multinomial(probs_b, num_samples=1) for probs_b in probs]).squeeze(-1)
        return token # [B, N]



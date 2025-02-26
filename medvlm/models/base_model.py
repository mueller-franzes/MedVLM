from pathlib import Path
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, Accuracy, AUROC
from torchmetrics.image import StructuralSimilarityIndexMeasure
# from torchmetrics.image import LearnedPerceptualImagePatchSimilarity  
from torchvision.utils import save_image
from einops import rearrange
import  torch.optim.lr_scheduler as lr_scheduler

from .utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot, tensor_mask2image
import wandb
import math 

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
        
class CLIPLossA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.temperature = nn.Parameter(torch.tensor([1.0]))
        logit_scale = math.log(1 / 0.07)
        self.logit_scale = nn.Parameter(logit_scale * torch.ones([])) 

    # def forward(self, image_embeddings, text_embeddings):
    #     self.logit_scale.data.clamp_(math.log(1), math.log(100))
    #     temperature = torch.exp(self.logit_scale)
    
    #     # https://github.com/moein-shariatnia/OpenAI-CLIP
    #     logits = temperature * text_embeddings @ image_embeddings.T 
    #     images_similarity = image_embeddings @ image_embeddings.T
    #     texts_similarity = text_embeddings @ text_embeddings.T
    #     targets = F.softmax(
    #         (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    #     )
    #     texts_loss = self.cross_entropy(logits, targets, reduction='none')
    #     images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
    #     loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    #     return loss.mean(), texts_loss.mean(), images_loss.mean()
    
    def forward(self, image_features, text_features, temperature=None):
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
        # https://github.com/lucidrains/CoCa-pytorch/blob/edee92c74e311ccfa4a0024412fd991c98aff5fd/coca_pytorch/coca_pytorch.py#L507
        self.logit_scale.data.clamp_(math.log(1), math.log(100))
        temperature = torch.exp(self.logit_scale) if temperature is None else temperature

        logits = temperature * image_features @ text_features.T
    
        device = logits.device
        labels = torch.arange(logits.shape[0], device=device, dtype=torch.long)


        images_loss = F.cross_entropy(logits, labels)
        texts_loss = F.cross_entropy(logits.T, labels)
        total_loss = (images_loss + texts_loss) / 2

        return total_loss, texts_loss, images_loss


    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


class BasicVLM(BasicModel):
    def __init__(
        self, 
        tokenizer_y,
        optimizer = torch.optim.AdamW,
        optimizer_kwargs ={'lr':5e-4},
        lr_scheduler= lr_scheduler.LinearLR, 
        lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':10000},
        save_hyperparameters=True
    ):
        super().__init__(optimizer, optimizer_kwargs, 
                        #  lr_scheduler, 
                        #  lr_scheduler_kwargs, 
                         save_hyperparameters=save_hyperparameters)

        self.tokenizer_y = tokenizer_y
        self.cliploss = CLIPLossA()

        # self.ce = nn.ModuleDict({state:CrossEntropy() for state in ["train_", "val_", "test_"]}) # 'train' not allowed as key
        # self.acc = nn.ModuleDict({state:Accuracy(task='multiclass', num_classes=3) for state in ["train_", "val_", "test_"]})
    
    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        x, y = batch['img'], batch['text']
        src_key_padding_mask = batch['src_key_padding_mask']
        self.batch_size = x.shape[0] 
        logging_dict = {}


        # ------------------- Run Model --------------------------
        logits  = self.forward(img=x, text=y[:, :-1], src_key_padding_mask=src_key_padding_mask) # [B, T, C]


        # ------------------------- Compute Loss ---------------------------
        ce = 0
        y = y[:, 1:] # Remove <SOS> 
        y_pred = logits.transpose(1, 2) # [B, N, C] ->   [B, C, N]
        ce = F.cross_entropy(y_pred, y, ignore_index=self.tokenizer_y.pad_token_id) 
       

        image_embeddings = self.memory_cls
        text_embeddings = self.tgt_cls
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        contrastive_loss, _, _ = self.cliploss(image_embeddings, text_embeddings)

        # Compute contrastive loss without temperature scaling
        with torch.no_grad():
            contrastive_loss_fix, texts_loss, images_loss  = self.cliploss(image_embeddings, text_embeddings, temperature=1)
        

        loss = ce+contrastive_loss#+self.loss2 #-cs+cs2 #self.loss2
        logging_dict['ce'] = ce
        logging_dict['ce2'] = contrastive_loss
        logging_dict['contastive_real'] = contrastive_loss_fix
        logging_dict['image2text'] = images_loss
        logging_dict['text2image'] = texts_loss
        logging_dict['temperature'] = self.cliploss.logit_scale.exp()
        # logging_dict['loss2'] = self.loss2
        logging_dict['loss'] = loss

        # --------------------- Compute Metrics  -------------------------------
        # pred_tokens = self.logits2tokens(logits) 
        with torch.no_grad():
            # num_tokens_y = y.size(1)
            # mask_padding_y = y == self.tokenizer_y.pad_token_id
            # pred_tokens_y = pred_tokens[:, -num_tokens_y:] 
            # logging_dict["acc"] = torch.sum(pred_tokens_y[~mask_padding_y] == y[~mask_padding_y])/torch.numel(y[~mask_padding_y])
  
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val,
                         batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=metric_name=="loss", sync_dist=True) 


        return logging_dict['loss'] 



    @torch.no_grad()
    def generate(self, x=None, y=None, x_pad_mask = None, return_logits=False, **kwargs):
        pred_tokens, logits = self._generate(x=x, y=y, x_pad_mask=x_pad_mask, **kwargs)
        pred = self.tokenizer_y.decode(pred_tokens)
        if return_logits:
            return pred, logits
        return pred
    

    @torch.no_grad()
    def _generate(self, x=None, y=None, x_pad_mask=None, max_new_tokens=None, batch_size=1, **kwargs):
        max_new_tokens = self.tokenizer_y.max_length if max_new_tokens is None else max_new_tokens 
        bos_token_id = self.tokenizer_y.bos_token_id
        eos_token_id = self.tokenizer_y.eos_token_id

        tokens = y
        if tokens is None:
            tokens = torch.tensor([bos_token_id], device=self.device).repeat(batch_size, 1)

        n=0
        while True:
            logits = self.forward(img=x, text=tokens, src_key_padding_mask=x_pad_mask) # TODO add src_key_padding_mask


            # attention_map = self.get_attention_maps()
            # source = x # [B, C, D, H, W]
            # b, c, *spatial_shape = source.shape
            # h = spatial_shape[1]//14
            # att_map = rearrange(attention_map, 'b (c d) (h w) -> b c d h w', c=c, h=h) 
            # att_map = F.interpolate(att_map, size=spatial_shape, mode='trilinear') # trilinear, area
            # save_image(tensor_cam2image(minmax_norm(source.cpu()), minmax_norm(att_map.cpu()), alpha=0.5), f"overlay.png", normalize=False)


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
        # return torch.argmax(logits, dim=-1)
        # TODO Something is wrong with this function 
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



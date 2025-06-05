import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformers import Dinov2Config, Dinov2Model
import x_transformers

from .mve_3D import MVE as MVE3D
from .mve_2D import MVE as MVE2D

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 

class Encoder(x_transformers.Encoder):
    def forward(self, 
            x, 
            mask=None, 
            src_key_padding_mask=None
        ):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x, None, src_key_padding_mask, None, mask)
    
class MST(nn.Module):
    def __init__(
        self, 
        out_ch=1, # no effect if return_last_hidden_layer=True
        backbone_type="dinov2",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "none", # transformer, linear, average, none 
        num_slices=32, # only relevant for slice_fusion_type=linear
        return_last_hidden_layer = True
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type
        self.return_last_hidden_layer = return_last_hidden_layer

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
            self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            self.backbone.mask_token = None  # Remove - otherweise unused parameters error"
            emb_ch = self.backbone.num_features
        elif backbone_type == "medinov2":
            # self.backbone = torch.hub.load('mueller-franzes/MDino', f'dinov2_vit{model_size}14', source='github')
            self.backbone = torch.hub.load('/home/homesOnMaster/gfranzes/Documents/code/MDino/', f'dinov2_vit{model_size}14', source='local')
            self.backbone.mask_token = None  # Remove - otherweise unused parameters error"
            emb_ch = self.backbone.num_features
        elif backbone_type == "dinov2-scratch":
            configuration = Dinov2Config()
            self.backbone = Dinov2Model(configuration)
            emb_ch = configuration.hidden_size
        elif backbone_type == "mve2D":
            self.backbone = MVE2D.load_from_checkpoint('runs/MVE/new/epoch=20-step=21000.ckpt')
            del self.backbone.decoder 
            emb_ch = self.backbone.emb_ch
        elif backbone_type == "mve3D":
            self.backbone = MVE3D.load_from_checkpoint('runs/MVE/3D/epoch=34-step=35000.ckpt')
            del self.backbone.decoder
            del self.backbone.mve2d.decoder
            emb_ch = self.backbone.emb_ch
        else:
            raise ValueError("Unknown backbone_type")


        self.emb_ch = emb_ch 
        if slice_fusion_type == "transformer":
            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=12 if emb_ch%12 == 0 else 8, 
                    dim_feedforward=1*emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    bias=False,
                ),
                num_layers=4,
                norm=nn.LayerNorm(emb_ch)
            )
            # self.slice_fusion = Encoder(
            #     dim = emb_ch,
            #     heads = 12 if emb_ch%12 == 0 else 8,
            #     ff_mult = 1,
            #     attn_dropout=0.0,
            #     pre_norm = True,
            #     depth = 4,
            #     attn_flash = True,
            #     ff_no_bias = True, 
            #     rotary_pos_emb=True,
            # )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type == 'linear':
            self.slice_fusion = nn.Linear(num_slices, 1)
        elif slice_fusion_type == 'average':
            pass 
        elif slice_fusion_type == "none":
            pass 
        else:
            raise ValueError("Unknown slice_fusion_type")

        if  not return_last_hidden_layer:
            self.linear = nn.Linear(emb_ch, out_ch)

        self.attention_maps_slice = []
        self.attention_maps = []



    def forward(self, x, src_key_padding_mask=None, save_attn=False):
        if save_attn:
            # fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            # torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()
    
        B, *_ = x.shape
        self.B = B 

        if self.backbone_type in ["mve3D"]:
            x = rearrange(x, 'b c d h w -> (b c) 1 d h w')
        else:
            x = rearrange(x, 'b c d h w -> (b c d) h w') #Flatten 3D volume into 2D slices
            x = x[:, None]

        if self.backbone_type in ["resnet", "dinov2"]:
            x = x.repeat(1, 3, 1, 1) # Gray to RGB
        # with autocast():
        # x = self.backbone(x)
        #TODO: specify whether training or not
        x = checkpoint(self.backbone, x.requires_grad_())
        # x = checkpoint(self.backbone.encode, x.requires_grad_())
        # x = self.backbone(x, is_training=True)['x_norm_patchtokens']

        if self.backbone_type == "dinov2-scratch":
            x = x.pooler_output

        if self.backbone_type in ["resnet", "dinov2"]:
            x = rearrange(x, '(b d) e -> b d e', b=B)
        elif self.backbone_type in ["mve2D"]:
            x = rearrange(x, '(b d) 1 e -> b d e', b=B)
        elif self.backbone_type in ["mve3D"]:
            x = rearrange(x, '(b c) k e -> b (c k) e', b=B)

        # x = rearrange(x, 'b c d h w -> (b c) 1 d h w')
        # x = checkpoint(self.backbone.encode, x)
        # x = rearrange(x, '(b c) l e -> b (c l) e', b=B)

        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            x = torch.concat([x, self.cls_token.repeat(B, 1, 1)], dim=1) # [B, 1+D, E]
            if src_key_padding_mask is not None: 
                src_key_padding_mask_cls = torch.zeros((B, 1), device=src_key_padding_mask.device, dtype=bool)
                src_key_padding_mask = torch.concat([src_key_padding_mask, src_key_padding_mask_cls], dim=1)
            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask) # [B, 1+D, L]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b e d')
            x = self.slice_fusion(x) # ->  [B, E, 1]
            x = rearrange(x, 'b e d -> b d e') #  ->  [B, 1, E]
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=True) #  [B, D, E] ->  [B, 1, E]


        if save_attn:
            # torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        if self.return_last_hidden_layer:
            return x 
        
        x = self.linear(x[:, -1])
        return x
    

    def forward_mask(self, block, x, mask_i):
        """
        Recompute the class token representation with the contribution of one patch token removed.
        """
        x = self.run_block(block, x, mask_i)
        x_norm = self.model.norm(x)
        x_norm_clstoken = x_norm[:, 0]
        pred = self.merge_slices(x_norm_clstoken)
        return pred
    


    def forward_attention(self, x):
        self.B, self.C, self.D, *_ = x.shape
        with torch.no_grad():
            x = self.reshape(x)
            x = self.backbone.prepare_tokens_with_masks(x)
            for blk in self.backbone.blocks[:-1]:
                x = blk(x)

            # Compute normal output 
            pred =  self.forward_mask(self.model.blocks[-1], x, None)
            
            # Compute last block with masked attention 
            token_relevance = []
            for c in range(self.C):
                for d in range(self.D):
                    for token_i in range(1, x.shape[1]):
                        pred_i = self.forward_mask(self.model.blocks[-1], x, (c, d, token_i))
                        rel_change = (pred.sigmoid() - pred_i.sigmoid()).abs() 
                        token_relevance.append(rel_change)

        token_relevance = torch.stack(token_relevance, dim=1)
        token_relevance = rearrange(token_relevance, 'B (C D M) K -> B C D M K', C=self.C, D=self.D)


        return pred, token_relevance
    

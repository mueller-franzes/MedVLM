import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformers import Dinov2Config, Dinov2Model

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 


class MST(nn.Module):
    def __init__(
        self, 
        out_ch=1, # no effect if return_last_hidden_layer=True
        backbone_type="dinov2",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "transformer", # transformer, linear, average, none 
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
        elif backbone_type == "dinov2-scratch":
            configuration = Dinov2Config()
            self.backbone = Dinov2Model(configuration)
            emb_ch = configuration.hidden_size


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


    def forward(self, x, src_key_padding_mask=None, save_attn=False):
        B, *_ = x.shape

        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # x = rearrange(x, 'b c d h w -> (b d) c h w')

        # if self.training:
        #     x = checkpoint(self.backbone, x)
        # else:
        #     x = self.backbone(x) # [(B D), C, H, W] -> [(B D), out] 
        x = checkpoint(self.backbone, x)
        # self.backbone(x, is_training=True)['x_norm_patchtokens']

        if self.backbone_type == "dinov2-scratch":
            x = x.pooler_output

        x = rearrange(x, '(b d) e -> b d e', b=B)

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



        if self.return_last_hidden_layer:
            return x 
        
        x = self.linear(x[:, 0])
        return x
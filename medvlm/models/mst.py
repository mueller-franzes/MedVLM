import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 


class MST(nn.Module):
    def __init__(
        self, 
        out_ch=1, 
        backbone_type="dinov2",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "transformer", # transformer, linear, average 
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
            self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            emb_ch = self.backbone.num_features

        self.emb_ch = emb_ch 
        if slice_fusion_type == "transformer":
            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=12, 
                    dim_feedforward=1*emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=1,
                norm=nn.LayerNorm(emb_ch)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type == 'linear':
            emb_ch = emb_ch*32
        elif slice_fusion_type == 'average':
            pass 
        elif slice_fusion_type == "none":
            pass 

        if  slice_fusion_type != "none":
            self.linear = nn.Linear(emb_ch, out_ch)


    def forward(self, x, src_key_padding_mask=None, save_attn=False):
        B, *_ = x.shape

        x = rearrange(x, 'b c d h w -> (b d c) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # x = self.backbone(x) # [(B D), C, H, W] -> [(B D), out] 
        x = checkpoint(self.backbone, x)
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)
            if src_key_padding_mask is not None: 
                src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                src_key_padding_mask = torch.concat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)# [Batch, L]
            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
            x = x[:, 0]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b (d e)')
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=False) #  [B, D, E] ->  [B, E]

        x = self.linear(x)

        return x
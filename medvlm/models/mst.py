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
        if save_attn:
            # fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            # torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()
    
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


        if save_attn:
            # torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        if self.return_last_hidden_layer:
            return x 
        
        x = self.linear(x[:, 0])
        return x
    

    
    def get_slice_attention(self):
        attention_map_slice = self.attention_maps_slice[-1] # [B, Heads, 1+D(+regs), 1+D(+regs)]
        attention_map_slice = attention_map_slice[:, :, 0, 1:] # [B, Heads, D]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True)

        # Average attention heads
        attention_map_slice = attention_map_slice.mean(dim=1)  #  [B, Heads, D] -> [B, D]
        attention_map_slice = attention_map_slice.view(-1) # [B*D]
        attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]
        return attention_map_slice
    
    def get_plane_attention(self):
        attention_map_dino = self.attention_maps[-1] # [B*D, Heads, 1+HW, 1+HW]
        num_register_tokens = self.backbone.num_register_tokens  if self.use_registers else 0
        img_slice = slice(num_register_tokens+1, None) 
        attention_map_dino = attention_map_dino[:,:, 0, img_slice] # [B*D, Heads, HW]
        attention_map_dino /= attention_map_dino.sum(dim=-1, keepdim=True)
        return attention_map_dino

    def get_attention_maps(self):
        attention_map_dino = self.get_plane_attention()
        attention_map_slice = self.get_slice_attention()
        attention_map = attention_map_slice*attention_map_dino
        return attention_map
    
    
    def register_hooks(self):
        # ------------------------- Backbone attention -----------------
        def enable_attention_dino(mod):
                forward_orig = mod.forward
                def forward_wrap(self2, x):
                    # forward_orig.__self__
                    B, N, C = x.shape
                    qkv = self2.qkv(x).reshape(B, N, 3, self2.num_heads, C // self2.num_heads).permute(2, 0, 3, 1, 4)
                    
                    q, k, v = qkv[0] * self2.scale, qkv[1], qkv[2]
                    attn = q @ k.transpose(-2, -1)
           
                    attn = attn.softmax(dim=-1)
                    attn = self2.attn_drop(attn)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self2.proj(x)
                    x = self2.proj_drop(x)

                    # Hook attention map 
                    self.attention_maps.append(attn)

                    return x
                
                mod.forward = lambda x: forward_wrap(mod, x)
                mod.foward_orig = forward_orig

        # Hook Dino Attention
        for name, mod in self.backbone.named_modules():
            if name.endswith('.attn'):
                enable_attention_dino(mod)

        # ------------------------- Slice fusion attention -----------------
        if self.slice_fusion_type != 'transformer':
            return 
        
        def enable_attention(module):
            forward_orig = module.forward
            def forward_wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            module.forward = forward_wrap
            module.foward_orig = forward_orig


        def append_attention_maps(module, input, output):
            self.attention_maps_slice.append(output[1])

        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.hooks.append(mod.register_forward_hook(append_attention_maps))


    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()

        # ------------------------- Backbone attention -----------------
        for name, mod in self.backbone.named_modules():
            if name.endswith('.attn'):
                mod.forward = mod.foward_orig
    
        # ------------------------- Slice fusion attention -----------------
        if self.slice_fusion_type != 'transformer':
            return 
        
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = mod.foward_orig

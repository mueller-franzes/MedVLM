import torch.nn as nn 
import torch 
import x_transformers
from einops import rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from math import sqrt

from .base_model import BasicModel

class Encoder(x_transformers.Encoder):
    def forward(self, 
            x, 
            mask=None, 
            src_key_padding_mask=None
        ):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x, None, src_key_padding_mask, None, mask)



class MVE(BasicModel):
    def __init__(self):
        super().__init__()


        emb_ch = 384
        heads = 6
        ff_mult = 2
        self.emb_ch = emb_ch

        self.encoder = Encoder(
            dim = emb_ch,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout=0.0,
            pre_norm = True,
            depth = 12,
            attn_flash = True,
            ff_no_bias = True, 
            rotary_pos_emb=True,
        )


        self.decoder = Encoder(
            dim = emb_ch,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout=0.0,
            pre_norm = True,
            depth = 12,
            attn_flash = True,
            ff_no_bias = True, 
            rotary_pos_emb=True,
        )

    

        patch_HW = 14 
        self.in_conv = nn.Conv2d(1, emb_ch, kernel_size=patch_HW, stride=patch_HW, padding=0, bias=False)
        self.out_conv = nn.ConvTranspose2d(emb_ch, 1, kernel_size=patch_HW, stride=patch_HW, padding=0, bias=False)

     

        self.enc_emb = nn.Parameter(torch.randn(1, 1, emb_ch)) # [1, L_e, emb_ch]
        self.dec_emb = nn.Parameter(torch.randn(1, 16*16, emb_ch)) # [1, L_e, emb_ch]
        


    def encode(self, img):
        x_in = self.in_conv(img) # [B, C, H, W] -> [B, emb_ch, H/patch, W/patch]
        x_flat = x_in.flatten(2)
        x_flat = x_flat.transpose(1, 2).contiguous()  # -> [B, H/patch*W/patch, emb_ch]
        B,L,E = x_flat.shape
        self.x_flat = x_flat

        pos_emb = self.dec_emb.expand(B, -1, -1)
        x_flat = x_flat+pos_emb


        cls_enc = self.enc_emb.expand(B, -1, -1) 
        num_hid_tok = cls_enc.size(1)
        x = torch.cat([x_flat, cls_enc], dim=1) # [B, 1+L, emb_ch]
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        mask[:, :-num_hid_tok] = False
        # z = self.encoder(x, mask=mask) # [B, 1+L, emb_ch]
        # z = self.encoder(x) # [B, 1+L, emb_ch]
        z = checkpoint(self.encoder, x, mask, None)
  
        z = z[:, -num_hid_tok:] # [B, L, 1]

        return z 


    def decode(self, z):
        B, _, E = z.shape
        pos_emb = self.dec_emb.expand(B, -1, -1)
        num_hid_tok =self.enc_emb.size(1)

        z = torch.cat([z, pos_emb], dim=1) # [B, 1+L, emb_ch]
        mask = torch.triu(torch.ones(z.size(1), z.size(1), device=z.device), diagonal=1).bool()
        # x = self.decoder(z, mask=mask) # [B, 1+L, emb_ch]
        # x = self.decoder(z) # [B, 1+L, emb_ch]
        x = checkpoint(self.decoder, z, mask, None)
    
        x = x[:, num_hid_tok:] # [B, L, emb_ch]

        self.ds = F.mse_loss(x, self.x_flat.detach())
        # x = (x +self.x_flat)-self.x_flat.detach()

        x = x.transpose(1, 2).contiguous() # -> [B, emb_ch, L]
        h = w =  int(sqrt(x.size(2)))
        x = x.reshape((B, E, h, w )) # [B, emb_ch, H/k, W/k]

        logits = self.out_conv(x) # [B, emb_ch, H/k, W/k] -> [B, 1, H, W]
        return logits



    def forward(self, img):
        z = self.encode(img)
        return z 
    
    def forward2(self, x):
        B, C, *_ = x.shape

        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None] # -> [BCD, 1, H, W]
        z = self.encode(x)
        x = self.decode(z)
        x = rearrange(x, '(b c d) 1 h w -> b c d h w', b=B, c=C)

        return x


        
        
     




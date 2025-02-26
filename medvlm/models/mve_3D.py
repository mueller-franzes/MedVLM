import torch.nn as nn 
import torch 
import x_transformers
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from .mve_2D import MVE as MVE2D

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
        self.emb_ch = emb_ch
        heads = 6
        ff_mult = 2

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


        self.mve2d = MVE2D.load_from_checkpoint('runs/MVE/new/epoch=47-step=48000.ckpt', strict=True) 
        for para in self.mve2d.parameters():
            para.requires_grad = False


        self.enc_emb = nn.Parameter(torch.randn(1, 32, emb_ch)) # [1, L_e, emb_ch]
        self.dec_emb = nn.Parameter(torch.randn(1, 32*8, emb_ch)) # [1, L_e, emb_ch]
        

    def encode(self, img):
        B, C, D, *_ = img.shape
        x = rearrange(img, 'b c d h w -> (b d) c h w')

        z = self.mve2d.encode(x) 
        self.target = z.detach() 
        x_flat = rearrange(z, '(b d) l e -> b (l d) e', d=D) 
        

        pos_emb = self.dec_emb.expand(B, -1, -1)
        x_flat = x_flat+pos_emb

     

        cls_enc = self.enc_emb.expand(B, -1, -1) 
        num_hid_tok = cls_enc.size(1)
        x = torch.cat([x_flat, cls_enc], dim=1) # [B, L+e, emb_ch]

        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        mask[:, :-num_hid_tok] = False
        # z = self.encoder(x, mask=mask) # [B, 1+L, emb_ch]
        # z = self.encoder(x) # [B, 1+L, emb_ch]
        z = checkpoint(self.encoder, x, mask, None)
  
        z = z[:, -num_hid_tok:] #  [B, 32, 384]
       

        # z, min_encoding_indices, loss = self.vq(z)
        self.vq_loss = 0 # loss.mean() 
        

        return z
    
    def decode(self, z, B, D):
        pos_emb = self.dec_emb.expand(B, -1, -1)
        z = torch.cat([z, pos_emb], dim=1) # [B, 32+32*8, emb_ch]
        mask = torch.triu(torch.ones(z.size(1), z.size(1), device=z.device), diagonal=1).bool()
        # x = self.decoder(z, mask=mask) # [B, 1+L, emb_ch]
        # x = self.decoder(z) # [B, 1+L, emb_ch]
        # x = self.encoder(z)
        x = checkpoint(self.decoder, z, mask, None)
    
        num_hid_tok = self.enc_emb.size(1)
        x = x[:, num_hid_tok:] # [B, L, emb_ch] [B, 32*8, 384]


        x_flat = rearrange(x, 'b (l d) e -> (b d) l e', d=D) # [B*32, 8, 384]
    
        img = self.mve2d.decode(x_flat)
        img = rearrange(img, '(b d) c h w -> b c d h w', b=B)
        return img 
    
    def forward(self, img):
        z = self.encode(img)
        return z 

    def forward2(self, img):
        B,C,D,*_ = img.shape 
        img = rearrange(img, 'b c d h w -> (b c) 1 d h w')
        z = self.encode(img)
        img = self.decode(z, B*C, D)
        img = rearrange(img, '(b c) 1 d h w -> b c d h w', b=B)
        return img



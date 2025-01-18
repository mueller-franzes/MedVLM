import torch
import torch.nn as nn 
from x_transformers import Encoder


torch.manual_seed(0)
emb_ch = 384
enc_multi = nn.TransformerEncoder(
    encoder_layer=nn.TransformerEncoderLayer(
        d_model=emb_ch,
        nhead=12 if emb_ch%12 == 0 else 8, 
        dim_feedforward=1*emb_ch,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    ),
    num_layers=4,
    norm=nn.LayerNorm(emb_ch)
)

torch.manual_seed(0)


model = Encoder(
    dim = emb_ch,
    heads = 12 if emb_ch%12 == 0 else 8,
    ff_mult = 1,
    attn_dropout=0.0,
    pre_norm = True,
    # rotary_pos_emb=False,
    depth = 4,
    attn_flash = True
)


x = torch.randn(2, 32, emb_ch)
src_key_padding_mask = torch.zeros_like(x).bool()[:,:,0]
tgt_size = x.size(1)
mask = torch.triu(torch.ones(tgt_size, tgt_size, device=x.device), diagonal=1).bool()
# mask = None 

output = model(x, mask = src_key_padding_mask, attn_mask=mask) 
print(output)
# output2 = enc_multi(x, mask=mask)
# print(output2)
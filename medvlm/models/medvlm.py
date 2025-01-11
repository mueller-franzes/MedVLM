import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
from typing import Optional
from torch import Tensor



class TransformerDecoder(nn.TransformerDecoder):
    # Add option to run foward pass without cross attention
    def forward_wo_cross_att(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: Optional[bool] = None) -> Tensor:
        output = tgt

        seq_len = nn.modules.transformer._get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = nn.modules.transformer._detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod.forward_wo_cross_att(output, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         tgt_is_causal=tgt_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    # Add option to run foward pass without cross attention
    def forward_wo_cross_att(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        # forward without encoder 
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size

        self.encoder = MST(slice_fusion_type='transformer')
        emb_ch = self.encoder.emb_ch 


        self.text_emb = nn.Embedding(vocab_size, emb_ch)
        nn.init.normal_(self.text_emb.weight, std=0.02)

        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=emb_ch,
                nhead=12 if emb_ch%12 == 0 else 8, 
                dim_feedforward=1*emb_ch,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=12,
            norm=nn.LayerNorm(emb_ch)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))

        # self.linear = nn.Linear(emb_ch, vocab_size, bias=False)
        # self.linear.weight = self.text_emb.weight 

    
    def forward(self, img, text=None, src_key_padding_mask=None):
        B = img.size(0)
        memory = self.encoder(img, src_key_padding_mask=src_key_padding_mask)
        
        text_emb = self.text_emb(text)
        text_emb = torch.concat([text_emb, self.cls_token.repeat(B, 1, 1)], dim=1) # [B, L+1, E]
        tgt_size = text_emb.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=text.device), diagonal=1).bool()
        tgt_key_padding_mask = text == self.tokenizer_y.pad_token_id
        tgt_key_padding_mask = torch.concat([tgt_key_padding_mask, torch.zeros((B, 1), device=text.device, dtype=bool)], dim=1)

        memory_mask = None 
        # memory_size = memory.size(1)-1 # without cls token 
        # memory_mask = torch.zeros(tgt_size, memory_size, device=memory.device).bool()
        # memory_mask[-1, :] = True # disable cross attention this way will result in NaN

        output = self.decoder(text_emb[:, :-1], memory[:, 1:], 
                              memory_mask = memory_mask, memory_key_padding_mask =src_key_padding_mask, 
                              tgt_mask=tgt_mask[:-1, :-1], tgt_key_padding_mask=tgt_key_padding_mask[:, :-1])
    
        # Ugly workaround to avoid NaN 
        output2 = self.decoder.forward_wo_cross_att(text_emb, 
                                                    tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        


        memory_cls = memory[:, 0] 
        tgt_cls = output2[:, -1]
        logits = output @ self.text_emb.weight.t()

        # Woraround to avoid need to return 
        self.memory_cls = memory_cls
        self.tgt_cls = tgt_cls

        return logits 

         

        
        
     
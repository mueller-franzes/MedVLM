import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 

class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size

        self.encoder = MST(slice_fusion_type='none')
        emb_ch = self.encoder.emb_ch 


        self.text_emb = nn.Embedding(vocab_size, emb_ch)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=emb_ch,
                nhead=12, 
                dim_feedforward=1*emb_ch,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=12,
            norm=nn.LayerNorm(emb_ch)
        )

        # self.linear = nn.Linear(emb_ch, vocab_size, bias=False)
        # self.linear.weight = self.text_emb.weight 

    
    def forward(self, img, text=None, src_key_padding_mask=None):
        memory = self.encoder(img, src_key_padding_mask=src_key_padding_mask)
        text_emb = self.text_emb(text)
        size = text.size(1)
        tgt_mask = torch.triu(torch.ones(size, size, device=text.device), diagonal=1).bool()
        tgt_key_padding_mask = text == self.tokenizer_y.pad_token_id
        output = self.decoder(text_emb, memory, 
                              memory_key_padding_mask =src_key_padding_mask, 
                              tgt_is_causal=True, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # output = self.linear(output)
        output = output @ self.text_emb.weight.t()
        return output 

         

        
        
     
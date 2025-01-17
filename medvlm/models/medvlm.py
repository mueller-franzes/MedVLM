import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
from typing import Optional
from torch import Tensor
# from transformers import AutoModel



class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size
        max_length = tokenizer_y.max_length

        self.encoder = MST(slice_fusion_type='none')
        for param in self.encoder.parameters():
            param.requires_grad = False
        emb_ch = self.encoder.emb_ch 


        self.text_emb = nn.Embedding(vocab_size, emb_ch)
        nn.init.normal_(self.text_emb.weight, std=0.02)
        self.text_pos_emb = nn.Embedding(max_length, emb_ch)

        # model = AutoModel.from_pretrained("GerMedBERT/medbert-512")
        # self.text_emb = model.embeddings.word_embeddings  # Embedding for tokenized words
        # self.text_pos_emb  = model.embeddings.position_embeddings  # Embedding for positions

        # self.decoder = TransformerDecoder(
        #     decoder_layer=TransformerDecoderLayer(
        #         d_model=emb_ch,
        #         nhead=12 if emb_ch%12 == 0 else 8, 
        #         dim_feedforward=1*emb_ch,
        #         dropout=0.0,
        #         batch_first=True,
        #         norm_first=True,
        #     ),
        #     num_layers=12,
        #     norm=nn.LayerNorm(emb_ch)
        # )

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
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


        self.cls_text = nn.Parameter(torch.randn(1, 1, emb_ch))

        self.cls_img = nn.Parameter(torch.randn(1, 1, emb_ch))

        # self.linear = nn.Linear(emb_ch, vocab_size, bias=False)
        # self.linear.weight = self.text_emb.weight 

        self.linear_a = nn.Linear(emb_ch, emb_ch, bias=False)
        self.linear_b = nn.Linear(emb_ch, emb_ch, bias=False)



    
    def forward(self, img, text=None, src_key_padding_mask=None):
        # src_key_padding_mask = torch.zeros_like(src_key_padding_mask).bool()
        B = img.size(0)
        memory = self.encoder(img, src_key_padding_mask=src_key_padding_mask)
        cls_img = self.cls_img.repeat(B, 1, 1)
        
        text_emb = self.text_emb(text) #[B, L] -> [B, L, E]
        text_emb += self.text_pos_emb(torch.arange(text.size(1), device=text.device))
        cls_text = self.cls_text.repeat(B, 1, 1)

        text_padding_mask = text == self.tokenizer_y.pad_token_id
        cls_padding_mask = torch.zeros((B, 1), device=text.device, dtype=bool)
        

        x = torch.cat([memory, cls_img, text_emb, cls_text], dim=1)
        src_padding_mask = torch.cat([src_key_padding_mask, cls_padding_mask, text_padding_mask, cls_padding_mask], dim=1)

        tgt_size = x.size(1)
        src_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=text.device), diagonal=1).bool()
        msize = memory.size(1)
        src_mask[:msize+1, :msize+1] = False
        src_mask[-1, :msize+1] = True 

        output = self.decoder(x, mask=src_mask, src_key_padding_mask=src_padding_mask)
        

        memory_cls = output[:, msize+1] 
        tgt_cls = output[:, -1] 
        logits = output[:, msize+1:-1]
        logits =  logits@ self.text_emb.weight.t()

        # memory_cls = memory[:, 0] 
        # tgt_cls = output2[:, -1]
        # logits = output @ self.text_emb.weight.t()

        # Woraround to avoid need to return 
        # self.memory_cls = memory_cls
        self.memory_cls = self.linear_a(memory_cls)
        # self.tgt_cls = tgt_cls
        self.tgt_cls = self.linear_b(tgt_cls)

        return logits 

         

        
        
     
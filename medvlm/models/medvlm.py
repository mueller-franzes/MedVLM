import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
from typing import Optional
from torch import Tensor
# from transformers import AutoModel
# from x_transformers import TransformerWrapper, Encoder


# class TransformerEncoder(Encoder):
#     def forward(self, x, context=None, src_key_padding_mask=None, context_mask=None, mask=None, self_attn_kv_mask=None, mems=None, mem_masks=None, seq_start_pos = None, cache = None, cache_age=1, return_hiddens=False, rotary_pos_emb=None, pos=None, context_pos=None, attn_bias=None, condition=None, in_attn_cond=None, layers_execute_order = None):
#         src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
#         mask = ~mask if mask is not None else None
#         return super().forward(x, context, src_key_padding_mask, context_mask, mask, self_attn_kv_mask, mems, mem_masks, seq_start_pos, cache, cache_age, return_hiddens, rotary_pos_emb, pos, context_pos, attn_bias, condition, in_attn_cond, layers_execute_order)


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

        # self.enc_vision = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=emb_ch,
        #         nhead=12 if emb_ch%12 == 0 else 8, 
        #         dim_feedforward=1*emb_ch,
        #         dropout=0.0,
        #         batch_first=True,
        #         norm_first=True,
        #     ),
        #     num_layers=4,
        #     norm=nn.LayerNorm(emb_ch)
        # )

        # self.enc_text = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=emb_ch,
        #         nhead=12 if emb_ch%12 == 0 else 8, 
        #         dim_feedforward=1*emb_ch,
        #         dropout=0.0,
        #         batch_first=True,
        #         norm_first=True,
        #     ),
        #     num_layers=4,
        #     norm=nn.LayerNorm(emb_ch)
        # )

        self.enc_multi = nn.TransformerEncoder(
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

        # self.enc_multi = TransformerEncoder(
        #     dim = emb_ch,
        #     heads = 12 if emb_ch%12 == 0 else 8,
        #     ff_mult = 1,
        #     attn_dropout=0.0,
        #     pre_norm = True,
        #     # sandwich_norm = True # https://arxiv.org/abs/2105.13290
        #     # rotary_pos_emb=True,
        #     # libi_pos_bias = True, # turns on ALiBi positional embedding
        #     depth = 4,
        #     attn_flash = True,
        #     # use_simple_rmsnorm = True
        #     ff_swish = True, # set this to True
        #     ff_glu = True,    # set to true to use for all feedforwards
        #     # ff_no_bias = True, # set to true to remove bias from feedforwards
        #     # attn_one_kv_head = True,

        #     # attn_qk_norm = True,       # https://arxiv.org/abs/2312.02696
        #     # attn_qk_norm_scale = 10    # new scale on the similarity, with groups of 1
        #     # attn_qk_norm_dim_scale = True # alternative to attn_qk_norm_scale
        # )



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


        # ----------------- Vision -----------------
        x = torch.cat([memory, cls_img], dim=1)
        src_padding_mask = torch.cat([src_key_padding_mask, cls_padding_mask], dim=1)
        memory = self.enc_multi(x, src_key_padding_mask=src_padding_mask)

        # ----------------- Text -----------------
        x = torch.cat([text_emb, cls_text], dim=1)
        src_padding_mask = torch.cat([text_padding_mask, cls_padding_mask], dim=1)
        tgt_size = x.size(1)
        src_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=text.device), diagonal=1).bool()
        text_emb = self.enc_multi(x,  mask=src_mask, src_key_padding_mask=src_padding_mask)

        # ----------------- Multi -----------------  
        x = torch.cat([memory[:, :-1], text_emb[:,:-1]], dim=1)
        src_padding_mask = torch.cat([src_key_padding_mask, text_padding_mask], dim=1)
        tgt_size = x.size(1)
        src_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=text.device), diagonal=1).bool()
        msize = memory.size(1)-1
        src_mask[:msize, :msize] = False
        multi_emb = self.enc_multi(x, mask=src_mask, src_key_padding_mask=src_padding_mask)
        

        memory_cls = memory[:, -1] 
        tgt_cls = text_emb[:, -1] 
        logits = multi_emb[:, msize:]
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

         

        
        
     
import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
import torch.nn.functional as F


class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size
        max_length = tokenizer_y.max_length

        # ----------------- Vision -----------------
        self.encoder = MST(slice_fusion_type='none')
        for param in self.encoder.backbone.parameters():
            param.requires_grad = False
        emb_ch = self.encoder.emb_ch 

        # ----------------- Text -----------------
        self.text_emb = nn.Embedding(vocab_size, emb_ch)
        nn.init.normal_(self.text_emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(max_length, emb_ch)

        self.multi_enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=emb_ch,
                nhead=12 if emb_ch%12 == 0 else 8, 
                dim_feedforward=1*emb_ch,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                bias=False,
                # activation=F.silu
            ),
            num_layers=8,
            norm=nn.LayerNorm(emb_ch)
        )

        self.cls_emb = nn.Parameter(torch.randn(1, 1, emb_ch))

        self.lm_cls = nn.Linear(emb_ch, emb_ch, bias=False)
        self.linear = nn.Linear(emb_ch, emb_ch, bias=False)
        

    def forward(self, img, text=None, src_key_padding_mask=None):
        B = img.size(0)
        cls_padding = torch.zeros((B, 1), device=img.device).bool()
        # ----------------- Vision -----------------
        memory_org = self.encoder(img, src_key_padding_mask=src_key_padding_mask)
        memory = memory_org + self.pos_emb(torch.arange(memory_org.size(1), device=text.device))
        memory = torch.cat([memory, self.cls_emb.repeat(B, 1, 1)], dim=1)
        src_key_padding_mask = torch.cat([src_key_padding_mask, cls_padding], dim=1)
        memory = self.multi_enc(memory, src_key_padding_mask=src_key_padding_mask)  
        
        # ----------------- Text -----------------
        text_emb = self.text_emb(text) #[B, L] -> [B, L, E]
        text_emb += self.pos_emb(torch.arange(text.size(1), device=text.device))
        text_emb = torch.cat([text_emb, self.cls_emb.repeat(B, 1, 1)], dim=1)

        text_padding_mask = text == self.tokenizer_y.pad_token_id
        text_padding_mask = torch.cat([text_padding_mask, cls_padding], dim=1)
        mask = torch.triu(torch.ones(text_emb.size(1), text_emb.size(1), device=text.device), diagonal=1).bool()
        # w_size = 2
        # mask = torch.ones(text_emb.size(1), text_emb.size(1), device=text.device)
        # mask = mask.triu(1).bool() | mask.tril(-w_size).bool()
        # mask[:, 0] = False # No mask for the first token, otherwise NaN for padding tokens 
        text_emb = self.multi_enc(text_emb, mask=mask, src_key_padding_mask=text_padding_mask)
        
        # ----------------- Multi -----------------  
        x = torch.cat([memory[:, :-1], text_emb[:, :-1]], dim=1)
        src_padding_mask = torch.cat([src_key_padding_mask[:, :-1], text_padding_mask[:, :-1]], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text.device), diagonal=1).bool()
        mask[:, :memory.size(1)-1] = False
        # w_size = 2
        # mask = torch.ones(x.size(1), x.size(1), device=text.device)
        # mask = mask.triu(1).bool() | mask.tril(-w_size).bool()
        # mask[:, :memory.size(1)] = False
        multi_output = self.multi_enc(x, mask=mask, src_key_padding_mask=src_padding_mask)


        x = torch.cat([text_emb[:, :-1], memory[:, :-1]], dim=1)
        src_padding_mask = torch.cat([text_padding_mask[:, :-1], src_key_padding_mask[:, :-1]], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text.device), diagonal=1).bool()
        mask[:, :text_emb.size(1)-1] = False
        multi_output2 = self.multi_enc(x, mask=mask, src_key_padding_mask=src_padding_mask)


        target = memory_org
        pred = multi_output2[:, text_emb.size(1)-1:]
        pred = self.linear(pred)

        # self.loss2 = F.mse_loss(pred, target)
        log_pred = F.log_softmax(pred, dim=-1)
        target = F.softmax(target, dim=-1)
        self.loss2 = -torch.sum(log_pred * target, dim=-1).mean()

        logits = multi_output[:, memory.size(1)-1:]
        logits = logits @ self.text_emb.weight.t()

        self.memory_cls = self.lm_cls(memory[:, -1] )
        self.tgt_cls = self.lm_cls(text_emb[:, -1] )

        return logits

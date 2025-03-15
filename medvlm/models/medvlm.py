import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from x_transformers import Encoder
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, Gemma3ForCausalLM
import math 
from .utils.functions import combine_padding_and_attention_masks
from tqdm import tqdm


class TransformerEncoder(Encoder):
    def forward(self, 
            x, 
            mask=None, 
            src_key_padding_mask=None,
        ):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask)



class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size
        max_length = tokenizer_y.max_length

        use_llama = False
        self.use_llama = use_llama

        # ----------------- Vision -----------------
        self.encoder = MST(backbone_type="dinov2", slice_fusion_type='none')
        for param in self.encoder.backbone.parameters():
            param.requires_grad = False
        emb_ch = self.encoder.emb_ch 

        self.vision_pos_emb = nn.Embedding(32*1*6, emb_ch)

        # ----------------- Text -----------------
        if not use_llama:
            # self.text_emb = nn.Embedding(vocab_size, emb_ch)
            # nn.init.normal_(self.text_emb.weight, std=0.02)
            # self.pos_emb = nn.Embedding(max_length, emb_ch)

            model = AutoModel.from_pretrained("GerMedBERT/medbert-512")
            for model.param in model.parameters():
                model.param.requires_grad = False
            self.text_emb = model.embeddings.word_embeddings
            self.pos_emb = model.embeddings.position_embeddings
            self.linear_proj = nn.Linear(self.text_emb.embedding_dim, emb_ch, bias=False)


        # ----------------- Multi -----------------
        if use_llama:
            # self.llama3 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
            self.llama3 = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.bfloat16) 
            for param in self.llama3.parameters():
                param.requires_grad = False
            self.lm_head =  self.llama3.lm_head # hidden to vocab size
            self.llama3.lm_head = nn.Identity()
            self.text_emb = self.llama3.model.embed_tokens
            self.linear_proj = nn.Linear(self.text_emb.embedding_dim, emb_ch, bias=False)

        else:
            config = {
                "dim": emb_ch,
                "heads": 12,
                "ff_mult": 1,
                "attn_dropout": 0.0,
                "pre_norm": True,
                "depth": 4,
                "attn_flash": True,
                "ff_no_bias": True,
                "rotary_pos_emb": True
            }
            self.multi_enc = TransformerEncoder(**config)

            # Seperate Encoders for Vision and Text
            # self.vis_enc = TransformerEncoder(**config)
            # self.text_enc = TransformerEncoder(**config)

        self.cls_emb = nn.Parameter(torch.randn(1, 1, emb_ch))
        self.bos_emb = nn.Parameter(torch.randn(1, 1, emb_ch))

        self.lm_cls = nn.Linear(emb_ch, emb_ch, bias=False)
        self.linear_vision = nn.Linear(emb_ch, emb_ch, bias=False)

        self._attention_maps_vlm = []
        self.save_attn = False 
        self.loss2 = 0

    
    def forward_vision(self, img, src_key_padding_mask=None, mask_vis_i=None):
        B = img.size(0)        
        cls_padding = torch.zeros((B, 1), device=img.device).bool()

        # ------------ Get Vision Embeddings ------------
        memory_org = self.encoder(img, src_key_padding_mask=src_key_padding_mask, save_attn=self.save_attn)
        self.memory_org = memory_org # save for later 
        
        # ------------ Prepare Vision Embeddings ------------
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros((B, memory_org.size(1)), device=img.device).bool() 
        self.src_key_padding_mask = src_key_padding_mask

        memory = memory_org + self.vision_pos_emb(torch.arange(memory_org.size(1), device=img.device))
        memory = torch.cat([memory, self.cls_emb.repeat(B, 1, 1)], dim=1)
        
        src_key_padding_mask = torch.cat([src_key_padding_mask, cls_padding], dim=1) 
        mask = torch.triu(torch.ones(memory.size(1), memory.size(1), device=memory.device), diagonal=1).bool()
        if mask_vis_i is not None:
            mask[mask_vis_i, :] = True
        
        # ------------ Vision ------------
        if self.use_llama:
            attention_mask = combine_padding_and_attention_masks(src_key_padding_mask, mask)
            x = memory @ self.linear_proj.weight  
            memory = self.llama3(inputs_embeds=x.to(self.llama3.dtype), attention_mask=attention_mask)["logits"]
            memory = self.linear_proj(memory.to(x.dtype))
        else:
            memory = self.multi_enc(memory, mask=mask, src_key_padding_mask=src_key_padding_mask) 
            # memory = checkpoint(self.multi_enc, memory.requires_grad_(), mask, src_key_padding_mask)
  

        memory_cls = self.lm_cls(memory[:, -1])
        # memory_cls = checkpoint(self.lm_cls, memory[:, -1])
        
        return memory_cls, memory[:, :-1]


    def forward_text(self, text):
        B = text.size(0)        
        cls_padding = torch.zeros((B, 1), device=text.device).bool()
    
        # ------------ Get Text Embeddings ------------
        text_emb = self.text_emb(text) #[B, L] -> [B, L, E]

        text_padding_mask = text == self.tokenizer_y.pad_token_id
        self.text_padding_mask = text_padding_mask
        text_padding_mask = torch.cat([text_padding_mask, cls_padding], dim=1)
        mask = torch.triu(torch.ones(text.size(1)+1, text.size(1)+1, device=text.device), diagonal=1).bool()

        if self.use_llama:
            cls_emb = self.cls_emb.repeat(B, 1, 1) @ self.linear_proj.weight 
            text_emb = torch.cat([text_emb, cls_emb ], dim=1)

            attention_mask = combine_padding_and_attention_masks(text_padding_mask, mask)
            text_emb = self.llama3(inputs_embeds=text_emb.to(self.llama3.dtype), attention_mask=attention_mask)["logits"]
            text_emb = self.linear_proj(text_emb.to(cls_emb.dtype))
            
        else:
            text_emb += self.pos_emb(torch.arange(text.size(1), device=text.device))
            text_emb = self.linear_proj(text_emb)
            text_emb = torch.cat([text_emb, self.cls_emb.repeat(B, 1, 1)], dim=1)

            text_emb = self.multi_enc(text_emb, mask, text_padding_mask)
            # text_emb = checkpoint(self.multi_enc, text_emb.requires_grad_(), mask, text_padding_mask)


        # Text Logits 
        tgt_cls = self.lm_cls(text_emb[:, -1])
        # tgt_cls = checkpoint(self.lm_cls, text_emb[:, -1])

        return tgt_cls, text_emb[:, :-1]
    

    def forward_vision_text(self, vision_emb, text_emb):
        x = torch.cat([vision_emb, text_emb], dim=1)
        src_padding_mask = torch.cat([self.src_key_padding_mask, self.text_padding_mask], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=vision_emb.device), diagonal=1).bool()

        if self.use_llama:
            x = x @ self.linear_proj.weight
            attention_mask = combine_padding_and_attention_masks(src_padding_mask, mask)
            multi_output = self.llama3(inputs_embeds=x.to(self.llama3.dtype), attention_mask=attention_mask)["logits"]
        else:
            multi_output = checkpoint(self.multi_enc, x, mask, src_padding_mask)


        logits = multi_output[:, vision_emb.size(1):]
        if self.use_llama:
            logits = self.lm_head(logits)
        else:
            logits = logits @ self.linear_proj.weight @ self.text_emb.weight.t()
        # logits = logits @ self.text_emb.weight.t()
        return logits 
    

    def forward_text_vision(self, text_emb, vision_emb):
        B = text_emb.size(0)   
        bos_emb = self.bos_emb.repeat(B, 1, 1)
        cls_padding = torch.zeros((B, 1), device=text_emb.device).bool()

        x = torch.cat([text_emb, bos_emb, vision_emb[:, :-1]], dim=1)
        src_padding_mask = torch.cat([self.text_padding_mask, cls_padding, self.src_key_padding_mask[:, :-1]], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text_emb.device), diagonal=1).bool()

        if self.use_llama:
            x = x @ self.linear_proj.weight
            attention_mask = combine_padding_and_attention_masks(src_padding_mask, mask)
            multi_output = self.llama3(inputs_embeds=x.to(self.llama3.dtype), attention_mask=attention_mask)["logits"]
            multi_output = self.linear_proj(multi_output.to(x.dtype))
        else:
            multi_output = checkpoint(self.multi_enc, x, mask, src_padding_mask)

        target = self.memory_org.detach()
        pred = multi_output[:, text_emb.size(1):]
        pred = self.linear_vision(pred)

        log_pred = F.log_softmax(pred, dim=-1)
        target = F.softmax(target, dim=-1)
        self.loss2 = -torch.sum(log_pred * target, dim=-1)
        self.loss2[src_padding_mask[:, text_emb.size(1):]] = 0
        self.loss2 = self.loss2.mean()

        return pred 

        

    def forward(self, img, text=None, src_key_padding_mask=None):    

        # ----------------- Vision -----------------
        memory_cls, memory = self.forward_vision(img, src_key_padding_mask=src_key_padding_mask)
        
        # ----------------- Text -----------------
        text_cls, text_emb = self.forward_text(text)


        # ----------------- Multi: Image -> Text -----------------  
        # text_logits = None 
        text_logits = self.forward_vision_text(memory, text_emb)
        

        # ----------------- Multi: Text -> Image -----------------  
        # vision_logits = None 
        vision_logits = self.forward_text_vision(text_emb, memory)

             
        return memory_cls, text_cls, text_logits, vision_logits
    

    

    def compute_similiarty(self, text, imgs, src_key_padding_masks, return_cls=False):
        # Vision 
        vision_cls, _ = self.forward_vision(imgs, src_key_padding_masks)
        vision_cls = F.normalize(vision_cls, dim=-1) # [B, 384]
        vision_cls = vision_cls.repeat_interleave(2, dim=0) # [2*B, 384]

        # Text
        text_cls, _ = self.forward_text(text) #[2, 1, 384]
        text_cls = F.normalize(text_cls, dim=-1) #[2, 384]
        text_cls = text_cls.repeat(imgs.size(0), 1) # [2*B, 384]

        # Calculate logits and probabilities for each image against both text prompts
        temperature = self.cliploss.logit_scale.exp()
        logits = torch.sum(vision_cls * text_cls, dim=-1).view(-1, 2) * temperature
        pred = logits.softmax(dim=-1)

        if return_cls:
            return pred, vision_cls, text_cls
        return pred 
    
    def compute_similiarty_attention(self, text, imgs, src_key_padding_masks):
        ref_pred, _, text_cls = self.compute_similiarty(text, imgs, src_key_padding_masks, True)

        # Calculate similarity while masking each slice of the image
        preds = [] 
        for slice_i in tqdm(range(math.prod(imgs.shape[1:3]))): #[C*D]
            vision_cls, _ = self.forward_vision(imgs, src_key_padding_masks, mask_vis_i=slice_i)
            vision_cls = F.normalize(vision_cls, dim=-1) # [B, 384]
            vision_cls = vision_cls.repeat_interleave(2, dim=0) # [2*B, 384]

            # Calculate logits and probabilities for each image against both text prompts
            temperature = self.cliploss.logit_scale.exp()
            logits = torch.sum(vision_cls * text_cls, dim=-1).view(-1, 2) * temperature
            pred = logits.softmax(dim=-1)
            preds.append(torch.abs(pred-ref_pred)[:, 0])
        
        saliency_map = torch.stack(preds, dim=1) # [B, C*D]
        saliency_map /= saliency_map.sum(dim=1)
        saliency_map = saliency_map.view(imgs.shape[0], *imgs.shape[1:3]) # [B, C, D]
        return ref_pred, saliency_map
    



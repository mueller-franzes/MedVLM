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
    def __init__(self, tokenizer_y, use_llm = False, only_cl=False):
        super().__init__(tokenizer_y=tokenizer_y, only_cl=only_cl)
        self.use_llm = use_llm # True: use pretrained LLM for text encoding AND multi-modal fusion 
        self.slice_pad_token_id = -1000


        # ----------------- Vision -----------------
        self.vision_encoder = MST(backbone_type="dinov2", slice_fusion_type='none')
        for param in self.vision_encoder.backbone.parameters():
            param.requires_grad = False
        emb_ch = self.vision_encoder.emb_ch 
        self.vision_pos_emb = nn.Embedding(32*1*6, emb_ch) # MAX: 32 Slices x 1 CLS Token x 6 Sequences (e.g T2, T1, Sub) 

        # ----------------- Text -----------------
        if not use_llm:
            # ------ Use unitialized text embeddings -------
            # self.text_encoder = nn.Embedding(vocab_size, emb_ch)
            # nn.init.normal_(self.text_encoder.weight, std=0.02)
            # self.pos_encoder = nn.Embedding(max_length, emb_ch)
            # self.text_vision_proj = nn.Linear(emb_ch, emb_ch, bias=False) # Just for compatibility: use frozen identity matrix
            # nn.init.eye_(self.text_vision_proj.weight)
            # self.text_vision_proj.weight.requires_grad = False

            # ------ Use pretrained BERT -------
            model = AutoModel.from_pretrained("GerMedBERT/medbert-512")
            for model.param in model.parameters():
                model.param.requires_grad = False
            self.text_encoder = model.embeddings.word_embeddings
            self.pos_encoder = model.embeddings.position_embeddings
            self.text_vision_proj = nn.Linear(self.text_encoder.embedding_dim, emb_ch, bias=False) # (eg. BERT 768 vs DINO-s 384)


        # ----------------- Multi -----------------
        if use_llm:
            self.multi_encoder = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
            # self.multi_encoder = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.bfloat16) 
            for param in self.multi_encoder.parameters():
                param.requires_grad = False
            self.text_encoder = self.multi_encoder.model.embed_tokens # Note: pos_encoder already included in text_encoderr 
            self.text_vision_proj = nn.Linear(self.text_encoder.embedding_dim, emb_ch, bias=False)
            self.lm_head =  self.multi_encoder.lm_head # last hidden layer output to logits
            self.multi_encoder.lm_head = nn.Identity() # Remove head to get last hidden layer output 

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
            self.multi_encoder = TransformerEncoder(**config)


        # ---------------- Other -----------------
        self.cls_emb = nn.Parameter(torch.randn(1, 1, emb_ch)) # classification token (text and vision)
        self.bos_emb = nn.Parameter(torch.randn(1, 1, emb_ch)) # beginning of sequence token (vision)

        self.cls_logits = nn.Linear(emb_ch, emb_ch, bias=False) # transform cls output to logits
        self.vision_logits = nn.Linear(emb_ch, emb_ch, bias=False) # transform last vision output to logits

        self._attention_maps_vlm = []
        self.loss2 = 0 # loss for text -> vision

    
    def forward_vision(self, img, mask_slice_i=None):
        B = img.size(0) # expect img to be of shape [B, C, D, H, W]       
        cls_padding = torch.zeros((B, 1), device=img.device).bool()

        # ------------ Get Vision Embeddings ------------
        vision_emb = self.vision_encoder(img) # [B, C, D, H, W] -> [B, C*D, E]
        self.vision_emb = vision_emb # save for later 
        
        # ------------ Prepare Embeddings and Masks ------------
        self.slice_padding_mask = img[:, :, :, 0, 0] == self.slice_pad_token_id # -> [B, C, D]
        self.slice_padding_mask = self.slice_padding_mask.view(B, -1) # -> [B, C*D]

        vision_emb = vision_emb + self.vision_pos_emb(torch.arange(vision_emb.size(1), device=img.device))
        vision_emb = torch.cat([vision_emb, self.cls_emb.repeat(B, 1, 1)], dim=1)
        
        vision_padding_mask = torch.cat([self.slice_padding_mask, cls_padding], dim=1) 
        mask = torch.triu(torch.ones(vision_emb.size(1), vision_emb.size(1), device=vision_emb.device), diagonal=1).bool()
        if mask_slice_i is not None:
            mask[mask_slice_i, :] = True
        
        # ------------ Vision ------------
        if self.use_llm:
            attention_mask = combine_padding_and_attention_masks(vision_padding_mask, mask)
            x = vision_emb @ self.text_vision_proj.weight  
            vision_emb = self.multi_encoder(inputs_embeds=x.to(self.multi_encoder.dtype), attention_mask=attention_mask)["logits"]
            vision_emb = self.text_vision_proj(vision_emb.to(x.dtype))
        else:
            vision_emb = self.multi_encoder(vision_emb, mask=mask, src_key_padding_mask=vision_padding_mask) 
            # vision_emb = checkpoint(self.multi_encoder, vision_emb.requires_grad_(), mask, vision_padding_mask)
  

        vision_cls = self.cls_logits(vision_emb[:, -1])
        # vision_cls = checkpoint(self.cls_logits, vision_emb[:, -1])
        
        return vision_cls, vision_emb[:, :-1]


    def forward_text(self, text):
        B = text.size(0)        
        cls_padding = torch.zeros((B, 1), device=text.device).bool()
    
        # ------------ Get Text Embeddings ------------
        text_emb = self.text_encoder(text) # [B, L] -> [B, L, E]

        text_padding_mask = text == self.tokenizer_y.pad_token_id
        self.text_padding_mask = text_padding_mask
        text_padding_mask = torch.cat([text_padding_mask, cls_padding], dim=1)
        mask = torch.triu(torch.ones(text.size(1)+1, text.size(1)+1, device=text.device), diagonal=1).bool()

        if self.use_llm:
            cls_emb = self.cls_emb.repeat(B, 1, 1) @ self.text_vision_proj.weight 
            text_emb = torch.cat([text_emb, cls_emb ], dim=1)

            attention_mask = combine_padding_and_attention_masks(text_padding_mask, mask)
            text_emb = self.multi_encoder(inputs_embeds=text_emb.to(self.multi_encoder.dtype), attention_mask=attention_mask)["logits"]
            text_emb = self.text_vision_proj(text_emb.to(cls_emb.dtype))
            
        else:
            text_emb += self.pos_encoder(torch.arange(text.size(1), device=text.device))
            text_emb = self.text_vision_proj(text_emb)
            text_emb = torch.cat([text_emb, self.cls_emb.repeat(B, 1, 1)], dim=1)

            text_emb = self.multi_encoder(text_emb, mask, text_padding_mask)
            # text_emb = checkpoint(self.multi_encoder, text_emb.requires_grad_(), mask, text_padding_mask)


        # Text Logits 
        tgt_cls = self.cls_logits(text_emb[:, -1])
        # tgt_cls = checkpoint(self.cls_logits, text_emb[:, -1])

        return tgt_cls, text_emb[:, :-1]
    

    def forward_vision_text(self, vision_emb, text_emb):
        x = torch.cat([vision_emb, text_emb], dim=1)
        src_padding_mask = torch.cat([self.slice_padding_mask, self.text_padding_mask], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=vision_emb.device), diagonal=1).bool()

        if self.use_llm:
            x = x @ self.text_vision_proj.weight
            attention_mask = combine_padding_and_attention_masks(src_padding_mask, mask)
            multi_output = self.multi_encoder(inputs_embeds=x.to(self.multi_encoder.dtype), attention_mask=attention_mask)["logits"]
        else:
            multi_output = checkpoint(self.multi_encoder, x, mask, src_padding_mask)


        multi_output = multi_output[:, vision_emb.size(1):]
        if self.use_llm:
            logits = self.lm_head(multi_output)
        else:
            logits = multi_output @ self.text_vision_proj.weight @ self.text_encoder.weight.t()
    
        return logits 
    

    def forward_text_vision(self, text_emb, vision_emb):
        B = text_emb.size(0)   
        bos_emb = self.bos_emb.repeat(B, 1, 1)
        cls_padding = torch.zeros((B, 1), device=text_emb.device).bool()

        x = torch.cat([text_emb, bos_emb, vision_emb[:, :-1]], dim=1)
        src_padding_mask = torch.cat([self.text_padding_mask, cls_padding, self.slice_padding_mask[:, :-1]], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text_emb.device), diagonal=1).bool()

        if self.use_llm:
            x = x @ self.text_vision_proj.weight
            attention_mask = combine_padding_and_attention_masks(src_padding_mask, mask)
            multi_output = self.multi_encoder(inputs_embeds=x.to(self.multi_encoder.dtype), attention_mask=attention_mask)["logits"]
            multi_output = self.text_vision_proj(multi_output.to(x.dtype))
        else:
            multi_output = checkpoint(self.multi_encoder, x, mask, src_padding_mask)

        
        pred = multi_output[:, text_emb.size(1):]
        pred = self.vision_logits(pred)

        # Workaround: required for loss calculation
        self._vision_padding_mask = src_padding_mask[:, text_emb.size(1):]

        return pred 

        

    def forward(self, img, text):    

        # ----------------- Vision -----------------
        memory_cls, memory = self.forward_vision(img)
        
        # ----------------- Text -----------------
        text_cls, text_emb = self.forward_text(text)


        # ----------------- Multi: Image -> Text -----------------  
        text_logits = None if self.only_cl else  self.forward_vision_text(memory, text_emb) 
        

        # ----------------- Multi: Text -> Image -----------------   
        vision_logits = None if self.only_cl else self.forward_text_vision(text_emb, memory)

             
        return memory_cls, text_cls, text_logits, vision_logits
    

    

    def compute_similiarty(self, text, imgs, return_cls=False):
        # Vision 
        vision_cls, _ = self.forward_vision(imgs)
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
    
    def compute_similiarty_attention(self, text, imgs):
        ref_pred, _, text_cls = self.compute_similiarty(text, imgs, return_cls=True)

        # Calculate similarity while masking each slice of the image
        preds = [] 
        for slice_i in tqdm(range(math.prod(imgs.shape[1:3]))): #[C*D]
            vision_cls, _ = self.forward_vision(imgs, mask_vis_i=slice_i)
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
    



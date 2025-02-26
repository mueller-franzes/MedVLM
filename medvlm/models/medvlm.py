import torch.nn as nn 
from .base_model import BasicVLM
from .mst import MST
import torch 
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from x_transformers import Encoder
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM

from .utils.functions import combine_padding_and_attention_masks



class TransformerEncoder(Encoder):
    def forward(self, 
            x, 
            mask=None, 
            src_key_padding_mask=None
        ):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x, None, src_key_padding_mask, None, mask)



class MedVLM(BasicVLM):
    def __init__(self, tokenizer_y):
        super().__init__(tokenizer_y=tokenizer_y)
        vocab_size = tokenizer_y.vocab_size
        max_length = tokenizer_y.max_length

        # ----------------- Vision -----------------
        self.encoder = MST(slice_fusion_type='none')
        # for param in self.encoder.backbone.parameters():
        #     param.requires_grad = False
        emb_ch = self.encoder.emb_ch 

        self.vision_pos_emb = nn.Embedding(32*1*6, emb_ch)

        # ----------------- Text -----------------
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
        # self.llama3 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        # for param in self.llama3.parameters():
        #     param.requires_grad = False
        # self.lm_head =  self.llama3.lm_head
        # self.llama3.lm_head = nn.Identity()
        # self.text_emb = self.llama3.model.embed_tokens
        # self.linear_proj = nn.Linear(self.text_emb.embedding_dim, emb_ch, bias=False)

        
        config = {
            "dim": emb_ch,
            "heads": 12,
            "ff_mult": 2,
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
        # self.bos_emb = nn.Parameter(torch.randn(1, 1, emb_ch))

        self.lm_cls = nn.Linear(emb_ch, emb_ch, bias=False)
        # self.linear_vision = nn.Linear(emb_ch, emb_ch, bias=False)

        self._attention_maps_vlm = []
        self.save_attn = False 
        

    def forward(self, img, text=None, src_key_padding_mask=None):
        B = img.size(0)        
        
        

        cls_padding = torch.zeros((B, 1), device=img.device).bool()
        # ----------------- Vision -----------------
        memory_org = self.encoder(img, src_key_padding_mask=src_key_padding_mask, save_attn=self.save_attn)
        
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros((B, memory_org.size(1)), device=img.device).bool() 
        # else:
        #     src_key_padding_mask =  src_key_padding_mask.repeat_interleave(memory_org.size(1)//src_key_padding_mask.size(1) , dim=1)


        #src_key_padding_mask = torch.zeros((B, memory_org.size(1)), device=img.device).bool()
        # memory = memory_org + self.linear_proj(self.pos_emb(torch.arange(memory_org.size(1), device=text.device)))
        memory = memory_org + self.vision_pos_emb(torch.arange(memory_org.size(1), device=text.device))


        memory = torch.cat([memory, self.cls_emb.repeat(B, 1, 1)], dim=1)
        
        src_key_padding_mask = torch.cat([src_key_padding_mask, cls_padding], dim=1)
        # memory = self.multi_enc(memory, src_key_padding_mask=src_key_padding_mask)  
        mask = torch.triu(torch.ones(memory.size(1), memory.size(1), device=memory.device), diagonal=1).bool()
        
        # memory = self.multi_enc(memory, mask=mask, src_key_padding_mask=src_key_padding_mask) 
        memory = checkpoint(self.multi_enc, memory, mask, src_key_padding_mask)
        
        # ----------------- Text -----------------
        text_emb = self.text_emb(text) #[B, L] -> [B, L, E]
        text_emb += self.pos_emb(torch.arange(text.size(1), device=text.device))
        text_emb = self.linear_proj(text_emb)
        text_emb = torch.cat([text_emb, self.cls_emb.repeat(B, 1, 1)], dim=1)

        text_padding_mask = text == self.tokenizer_y.pad_token_id
        text_padding_mask = torch.cat([text_padding_mask, cls_padding], dim=1)
        mask = torch.triu(torch.ones(text_emb.size(1), text_emb.size(1), device=text.device), diagonal=1).bool()
        # w_size = 2
        # mask = torch.ones(text_emb.size(1), text_emb.size(1), device=text.device)
        # mask = mask.triu(1).bool() | mask.tril(-w_size).bool()
        # mask[:, 0] = False # No mask for the first token, otherwise NaN for padding tokens 
        # text_emb = self.multi_enc(text_emb, mask=mask, src_key_padding_mask=text_padding_mask)
        text_emb = checkpoint(self.multi_enc, text_emb, mask, text_padding_mask)

        if self.save_attn:
            # fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            # torch.backends.mha.set_fastpath_enabled(False)
            self._attention_maps_vlm = []
            self.hooks = []
            self.register_hooks()
    

        # ----------------- Multi: Image -> Text -----------------  
        x = torch.cat([memory[:, :-1], text_emb[:, :-1]], dim=1)
        src_padding_mask = torch.cat([src_key_padding_mask[:, :-1], text_padding_mask[:, :-1]], dim=1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text.device), diagonal=1).bool()
        # mask[:, :memory.size(1)-1] = False
        # w_size = 2
        # mask = torch.ones(x.size(1), x.size(1), device=text.device)
        # mask = mask.triu(1).bool() | mask.tril(-w_size).bool()
        # mask[:, :memory.size(1)] = False
        # multi_output = self.multi_enc(x, mask=mask, src_key_padding_mask=src_padding_mask)
        multi_output = checkpoint(self.multi_enc, x, mask, src_padding_mask)

        # attention_mask = combine_padding_and_attention_masks(src_padding_mask, mask)
        # x = self.linear_proj.weight @ x 
        # multi_output = self.llama3(inputs_embeds=x, attention_mask=attention_mask)["logits"]

        logits = multi_output[:, memory.size(1)-1:]
        logits = logits @ self.linear_proj.weight @ self.text_emb.weight.t()
        # logits = logits @ self.text_emb.weight.t()
        

        # # ----------------- Multi: Text -> Image -----------------  
        # x = torch.cat([text_emb[:, :-1], self.bos_emb.repeat(B, 1, 1), memory[:, :-2]], dim=1)
        # src_padding_mask = torch.cat([text_padding_mask[:, :-1], cls_padding, src_key_padding_mask[:, :-2]], dim=1)
        # mask = torch.triu(torch.ones(x.size(1), x.size(1), device=text.device), diagonal=1).bool()
        # # mask[:, :text_emb.size(1)-1] = False
        # multi_output2 = self.multi_enc(x, mask=mask, src_key_padding_mask=src_padding_mask)
        # # multi_output2 = checkpoint(self.multi_enc, x, mask, src_padding_mask, None)

        # target = memory_org.detach()
        # pred = multi_output2[:, text_emb.size(1)-1:]
        # pred = self.linear_vision(pred)

        # # self.loss2 = F.mse_loss(pred, target)
        # log_pred = F.log_softmax(pred, dim=-1)
        # target = F.softmax(target, dim=-1)
        # self.loss2 = -torch.sum(log_pred * target, dim=-1)
        # self.loss2[src_padding_mask[:, text_emb.size(1)-1:]] = 0
        # self.loss2 = self.loss2.mean()

        # ----------------- Contrastive embeddings  -----------------
        self.memory_cls = self.lm_cls(memory[:, -1] )
        self.tgt_cls = self.lm_cls(text_emb[:, -1] )

        
        if self.save_attn:
            # torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        # return None 
        return logits



    def get_vlm_attention(self):
        attention_map_slice = self._attention_maps_vlm[-1] # [B, Heads, 1+D(+regs), 1+D(+regs)]
        attention_map_slice = attention_map_slice[:, :, -1, :] # [B, Heads, D]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True)

        # Average attention heads
        attention_map_slice = attention_map_slice.mean(dim=1)  #  [B, Heads, D] -> [B, D]
        # attention_map_slice = attention_map_slice.view(-1) # [B*D]
        # attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]
        return attention_map_slice
    
    def get_vision_attention(self):
        return self.encoder.get_attention_maps()

    def get_attention_maps(self):
        attention_map_dino = self.get_vision_attention() # [B, D, HW]
        attention_map_slice = self.get_vlm_attention() # [B, N]
        slices = attention_map_dino.size(1)
        slice_attention = attention_map_slice[:, :slices]
        text_attention = attention_map_slice[:, slices:]
        image_attention = slice_attention[:,:,None]*attention_map_dino
        
        return image_attention
    
    
    def register_hooks(self):
        def enable_attention(module):
            forward_orig = module.forward
            def forward_wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            module.forward = forward_wrap
            module.foward_orig = forward_orig

        def append_attention_maps(module, input, output):
            self._attention_maps_vlm.append(output[1])

        for _, mod in self.multi_enc.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.hooks.append(mod.register_forward_hook(append_attention_maps))


    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()
    
        for _, mod in self.multi_enc.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = mod.foward_orig
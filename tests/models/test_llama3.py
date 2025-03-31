import torch

from medvlm.models.tokenizer import Tokenizer
from transformers import LlamaForCausalLM

device=torch.device('cuda')

tokenizer = Tokenizer()

text = "Please write a report for the following patient"
text = tokenizer.tokenizer(
            text,
            add_special_tokens=True,  # Avoid automatic addition of special tokens
            return_tensors='pt',
            padding=False,
            truncation=True
        ).input_ids[0]

llama3 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16)
llama3.to(device)
input_ids = text[None].to(device)

text_emb = llama3.model.embed_tokens

while True:
    with torch.no_grad():
        inputs_embeds = text_emb(input_ids)
        # logits = llama3(input_ids=input_ids).logits
        logits = llama3(inputs_embeds=inputs_embeds).logits
        
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    if next_token_id == tokenizer.eos_token_id:
        break
    print(tokenizer.decode(input_ids)[0])


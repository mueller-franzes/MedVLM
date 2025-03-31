import torch

from medvlm.models.tokenizer import Tokenizer
from transformers import Gemma3ForCausalLM 

device=torch.device('cuda')

tokenizer = Tokenizer(model_name="google/gemma-3-1b-pt")

text = "Please write a report for the following patient"
tokens = tokenizer.tokenizer(
            text,
            add_special_tokens=True,  # Avoid automatic addition of special tokens
            return_tensors='pt',
            padding=False,
            truncation=True
        ).input_ids[0]
tokens = tokens[None].to(device)


model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.bfloat16) 
model.to(device)

text_emb = model.model.embed_tokens

while True:
    with torch.no_grad():
        inputs_embeds = text_emb(tokens)
        logits = model(inputs_embeds=inputs_embeds).logits
        
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token_id], dim=-1)
    if next_token_id == tokenizer.eos_token_id:
        break
    print(tokenizer.decode(tokens)[0])


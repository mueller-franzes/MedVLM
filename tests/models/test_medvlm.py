import torch
from transformers import AutoTokenizer 
from medvlm.models.medvlm import MedVLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

input = torch.randn((1, 1, 32, 224, 224))
text = "Das ist ein Test"
text_tokens = tokenizer(text, return_tensors="pt").input_ids # [B, 7]

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# text_tokens = tokenizer(
#     text,
#     padding="max_length",  # Pad to the maximum length
#     truncation=True,       # Truncate if the text is longer than 512 tokens
#     max_length=512,        # Set the fixed length to 512 tokens
#     return_tensors="pt"    # Return PyTorch tensors (optional)
# )

device=torch.device('cuda')

model = MedVLM()

model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6
print("Number Parameters", num_params)

pred = model(input.to(device), text=text_tokens.to(device))
print(pred.shape)

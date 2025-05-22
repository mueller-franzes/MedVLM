import torch
from medvlm.models.tokenizer import Tokenizer
from medvlm.models.medvlm import MedVLM
from medvlm.utils.gpu import get_gpu_with_max_free_memory


# ---- Pseudo image ---------
input = torch.randn((1, 3, 32, 224, 224))

# ---- Random text ---------
tokenizer = Tokenizer() # model_name="meta-llama/Llama-3.2-1B"
text = "Das ist ein Test"
text_tokens = tokenizer(text)[None] # [B, 7]


# ---- Model ---------
model = MedVLM(tokenizer_y=tokenizer)
best_gpu_index, max_free_memory = get_gpu_with_max_free_memory()
device=torch.device(f'cuda:{best_gpu_index}')
model.to(device)

# ---- Number of parameters ---------
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6
print("Number Parameters", num_params)

# ---- Forward pass ---------
pred = model(input.to(device), text=text_tokens.to(device))
# print(pred.shape)

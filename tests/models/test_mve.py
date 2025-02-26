import torch

from medvlm.models.mve_2D import MVE

input = torch.randn((1, 1, 224, 224))

device=torch.device('cuda')

model = MVE()

model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6
print("Number Parameters", num_params)

pred = model(input.to(device))
print(pred.shape)

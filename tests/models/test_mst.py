import torch

from medvlm.models.mst import MST

input = torch.randn((1, 1, 32, 224, 224))

device=torch.device('cuda')

model = MST(
    # backbone_type="dinov2",
    # slice_fusion_type='transformer',
)

model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6
print("Number Parameters", num_params)

pred = model(input.to(device), save_attn=False)
print(pred.shape)

import torch
from pathlib import Path 
from torchvision.utils import save_image


from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D
from medvlm.models.utils.functions import tensor2image
from medvlm.models.mve_2D import MVE as MVE2D
from medvlm.models.mve_3D import MVE as MVE3D



ds = UKA_Dataset3D(split='test', tokenizer=None)

device=torch.device('cuda')

# model = MVE3D.load_from_checkpoint('runs/MVE/3D/epoch=34-step=35000.ckpt')
model = MVE2D.load_from_checkpoint('runs/MVE/new/epoch=20-step=21000.ckpt')
model.to(device)

item = ds[100]
img = item['img'][None]

pred = model.forward2(img.to(device)).cpu()

print("MSE", torch.mean((pred-img)**2))

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
save_image(tensor2image(img) , path_out/'original.png', normalize=True)
save_image(tensor2image(pred.cpu()) , path_out/'prediction.png', normalize=True)

import torch 
from pathlib import Path 
from torchvision.utils import save_image

from medvlm.models.utils.functions import tensor2image, tensor_mask2image, one_hot
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D


ds = UKA_Dataset3D(
    split='train',
    # flip=True, 
    # noise=True, 
    # random_center=True, 
    # random_rotate=True,
    # use_s3=True
)

print("Dataset Length", len(ds))

item = ds[100]
uid = item["uid"]
img = item['img']
label = item['label']
src_key_padding_mask = item['src_key_padding_mask']
print(img.min(), img.max(), img.mean(), img.std())

print("UID", uid, "Image Shape", list(img.shape), "Label", label, "src_key_padding_mask", src_key_padding_mask)

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
save_image(tensor2image(img[None]) , path_out/'test.png', normalize=True)

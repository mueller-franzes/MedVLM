import torch 
from pathlib import Path 
from torchvision.utils import save_image

from medvlm.models.utils.functions import tensor2image, tensor_mask2image, one_hot
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D


ds = CTRATE_Dataset3D(
    split='train',
    # flip=True, 
    # noise=True, 
    # random_center=True, 
    # random_rotate=True,
    # use_s3=True
)

print("Dataset Length", len(ds))

# item = ds[10]
item = ds.get_examuid('train_1_a')
uid = item["uid"]
img = item['img']
label = item['label']
print(img.min(), img.max(), img.mean(), img.std())

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
save_image(tensor2image(img[None]) , path_out/'test.png', normalize=True)
# save_image(tensor2image(item['mask'][None]), path_out/'mask.png', normalize=True)
# save_image(tensor_mask2image(img[None], one_hot(item['mask'], 3), alpha=0.25), path_out/'overlay.png', normalize=False)
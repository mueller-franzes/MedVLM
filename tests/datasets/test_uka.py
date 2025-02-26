import torch 
from pathlib import Path 
from torchvision.utils import save_image
import pandas as pd 
from tqdm import tqdm

from medvlm.models.utils.functions import tensor2image, tensor_mask2image, one_hot
from medvlm.data.datasets.dataset_3d_uka import UKA_Dataset3D


ds = UKA_Dataset3D(
    split='train',
    random_flip=True, 
    random_noise=True, 
    random_center=True, 
    random_rotate=True,
    random_inverse=True,
    # use_s3=True
)

# results = []
# for path in tqdm((ds.path_root/'data_unilateral').iterdir()):
#     files = {path.name.split('.')[0]: 1 for path in path.iterdir()}
#     results.append({'UID': path.stem, **files})
# df = pd.DataFrame(results)
# print("Dataset Length", len(ds))

item = ds[20]
uid = item["uid"]
img = item['img']
label = item['label']
src_key_padding_mask = item['src_key_padding_mask']
print(img.min(), img.max(), img.mean(), img.std())

print("UID", uid, "Image Shape", list(img.shape), "Label", label, "src_key_padding_mask", src_key_padding_mask)

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
save_image(tensor2image(img[None]) , path_out/'test.png', normalize=True)

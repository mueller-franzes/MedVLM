import torch
import gc
from medvlm.data.datasets.dataset_3d_ctrate import CTRATE_Dataset3D
import pandas as pd
from tqdm import tqdm

def print_mem(stage=""):
    device = torch.device("cuda:3")
    allocated = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    print(f"[{stage}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


def bytes_to_mb(x): return x / (1024 * 1024)

def print_large_tensors_with_names(model, min_size_mb=1.0, device='cuda:3'):
    print(f"Scanning for tensors > {min_size_mb} MB on {device}...\n")

    # Step 1: Build a map from id(param) -> name
    param_name_map = {id(p): name for name, p in model.named_parameters()}

    found = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj.data if hasattr(obj, 'data') else obj
                if str(tensor.device) == device:
                    size_bytes = tensor.element_size() * tensor.nelement()
                    size_mb = bytes_to_mb(size_bytes)
                    if size_mb >= min_size_mb:
                        obj_id = id(obj)
                        name = param_name_map.get(obj_id, "<unnamed>")
                        found.append((size_mb, tensor.shape, type(obj), name))
        except Exception:
            continue

    # Sort by size descending
    found.sort(reverse=True)

    for size_mb, shape, obj_type, name in found:
        print(f"{obj_type.__name__:>20} | {size_mb:6.2f} MB | shape={tuple(shape)} | name={name}")

    print(f"\nTotal large tensors found: {len(found)}")

def get_mask_max(ds, examuid):
    idx = ds.item_pointers.index(examuid)
    examuid = ds.item_pointers[idx] # Note: one exam can have multiple reconstructions 
    item = ds.df.loc[examuid]
    folder, patientid, examid = examuid.split('_')
    subfolder = f'{folder}_{patientid}'
    uid = f'{examuid}_{1}'
    path_item = ds.path_root_prep/'data'/folder/subfolder/examuid
    # img = ds.load_img(path_item/f'{uid}.nii.gz')
    seg_lung = ds.load_map(path_item/f'seg_lung.nii.gz')
    img = ds.load_img(path_item/f'{uid}.nii.gz')
    tensor = seg_lung.data
    tensor_img = img.data
    tensor = tensor.squeeze(0)
    coords = tensor.nonzero(as_tuple=False)
    shape = torch.tensor(tensor.shape, dtype=torch.float32)
    min_coords = coords.min(dim=0).values.float()/shape
    max_coords = coords.max(dim=0).values.float()/shape
    del seg_lung, img, tensor, tensor_img
    return min_coords, max_coords


def find_faulty_samples():
    split = 'test' #'train'
    ds = CTRATE_Dataset3D(
        split=split,
        # flip=True, 
        # noise=True, 
        # random_center=True, 
        # random_rotate=True,
        # use_s3=True
    )

    global_min = torch.tensor([float('inf')] * 3)
    global_max = torch.tensor([-float('inf')] * 3)
    faulty_files = []
    df = pd.DataFrame(columns=["VolumeName","UID","ExamUID","PatientUID","Split","Fold"])
    for index in tqdm(ds.item_pointers, total=len(ds.item_pointers)):
        try:
            min_coords_temp, max_coords_temp = get_mask_max(ds, index)
            global_min = torch.min(global_min, min_coords_temp)
            global_max = torch.max(global_max, max_coords_temp)
            # print(global_min, global_max)
            split = index.split('_')
            row = [f"{index}_1.nii.gz", f"{index}_1", index, f"{split[0]}_{split[1]}", "test", '0']
            df.loc[len(df)]=row
            # if len(df)>300:
            #     break
        except RuntimeError:
            faulty_files.append(index)
            

    # print(faulty_files)
    path = '/hpcwork/p0021834/datasets/CT-RATE/'

    df.to_csv(path+'download/valid_not_defect.csv', index=False)
    # VolumeName,UID,ExamUID,PatientUID,Split,Fold

# find_faulty_samples()
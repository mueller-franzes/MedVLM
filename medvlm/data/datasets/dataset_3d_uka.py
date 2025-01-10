from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  
from torchvision import transforms as T
import torch.nn as nn

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, ZNormalization, CropOrPad

class UKA_Dataset3D(data.Dataset):
    PATH_ROOT = Path.home()/'Documents/datasets/ODELIA/UKA_all'
    LABEL = 'Karzinom' # Fibroadenom Adenose Lympfknoten Zyste Fettgewebsnekrose Hyperplasie Duktektasie Papillom Hamartom Radiäre Narbe Duktales Karzinom in situ (DCIS) Karzinom

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            random_flip = False,
            random_center=False,
            random_rotate=False,
            random_inverse=False,
            random_noise=False, 
            to_tensor = True,
            tokenizer = None,
        ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.tokenizer = tokenizer 
        self.split = split
        

        if transform is None: 
            self.transform = tio.Compose([
                tio.Flip((1,0)), # Just for viewing, otherwise upside down
                CropOrPad((224, 224, 32), random_center=random_center, padding_mode='minimum'), # WANRING: Padding mode also for LabelMap

                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)), 

                tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if random_flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if random_noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform

        
        # self.aug_2d = T.Compose([
        #     T.RandomCrop((224, 224), pad_if_needed=True, padding_mode='edge') if random_center else T.CenterCrop((224, 224)),

        #     T.Lambda(lambda x: x.transpose(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else T.Lambda(lambda x: x),
        #     T.RandomHorizontalFlip() if random_flip else nn.Identity(),
        #     T.RandomVerticalFlip() if random_flip else nn.Identity(),
        #     T.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x) if random_inverse else T.Lambda(lambda x: x),
            
        #     GaussianNoise(sigma=(0.0, 0.25), clip=False) if noise else T.Lambda(lambda x: x),
        # ])
 

        # Get split file 
        path_csv = self.path_root/f'metadata/split_regex_{self.LABEL}.csv' 
        self.df = self.load_split(path_csv, fold=fold, split=split, fraction=fraction)
        

        path_report = self.path_root/f'metadata/mrt-befund.csv' 
        df_reports = pd.read_csv(path_report)
    
        df_left = df_reports[['AnforderungsNr', 'UntersuchungsNr', 'Linke Mamma']]
        df_left = df_left.rename(columns={'Linke Mamma':'Findings'})
        df_left = df_left[df_left['Findings'].notna()]
        df_left.insert(0, 'UID', df_left['AnforderungsNr'].astype(str)+df_left['UntersuchungsNr'].astype(str).str.zfill(2)+'_left')
        df_left = df_left.drop_duplicates(subset="UID", keep="first")
        df_right = df_reports[['AnforderungsNr', 'UntersuchungsNr', 'Rechte Mamma']]
        df_right = df_right.rename(columns={'Rechte Mamma':'Findings'})
        df_right = df_right[df_right['Findings'].notna()]
        df_right.insert(0, 'UID', df_right['AnforderungsNr'].astype(str)+df_right['UntersuchungsNr'].astype(str).str.zfill(2)+'_right')
        df_right = df_right.drop_duplicates(subset="UID", keep="first")

        df_reports = pd.concat([df_left, df_right]).set_index('UID')
        

        # if tokenizer is not None:
        #     eos_token_id = tokenizer.tokenizer.eos_token_id
        #     df_reports = df_reports[df_reports['Findings'].apply(lambda x: (tokenizer(x)[0] == eos_token_id).nonzero().squeeze().item()<511)] # only 96 cases 161

        self.df_reports = df_reports
        # df_reports[df_reports.index.isin(self.df['UID'])]['Findings'].isna().sum()
        self.df = self.df[self.df['UID'].isin(df_reports.index)]

        self.item_pointers = self.df.index.tolist()

    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        label = item[self.LABEL]       
        uid = item['UID']
        text = self.df_reports.loc[uid]['Findings']
 

        img = self.load_img(self.path_root/'data_unilateral'/uid/'Sub.nii.gz')
        # img = self.load_img([self.path_root/'data_unilateral'/uid/img_name for img_name in ['Sub.nii.gz', 'Pre.nii.gz', 'T2_resampled.nii.gz']])
        mask_fg = img.data[0].sum(0).sum(0)>0
        mask_fg = tio.LabelMap(tensor=torch.ones_like(img.data, dtype=torch.int32)*mask_fg[None] , affine=img.affine)
        sub = tio.Subject(img=img, mask_fg=mask_fg)
    
        sub = self.transform(sub)

        img = sub['img']
        mask_fg = sub['mask_fg']
        src_key_padding_mask = ~(mask_fg[0].sum(-1).sum(-1)>0)

        # if (self.split == 'train'): 
        #     rand_idx = torch.randperm(32)
        #     img = img[:, rand_idx]
        #     src_key_padding_mask = src_key_padding_mask[rand_idx]

        # for slice_n in range(img.shape[1]):
        #     if src_key_padding_mask[slice_n]:
        #         continue
        #     img[:, slice_n] = self.aug_2d(img[:, slice_n].clone())# WARNING: without .clone() unexpected behavior for .moveaxis() 

        if self.tokenizer is not None:
            text = self.tokenizer(text)

        # img[:, src_key_padding_mask] = -1000

  
        return {'uid':uid, 'img':img , 'text':text, 'label':label, 'src_key_padding_mask':src_key_padding_mask}



    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
    
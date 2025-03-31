from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  
from torchvision import transforms as T
import torch.nn as nn

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, ZNormalization, CropOrPad, RescaleIntensity

class UKA_Dataset3D(data.Dataset):
    # PATH_ROOT = Path.home()/'Documents/datasets/ODELIA/UKA_all'
    # PATH_ROOT = Path('/mnt/ocean_storage/users/gfranzes/UKA_Breast/')
    PATH_ROOT = Path('/mnt/datasets_gustav/UKA/')

    LABEL = 'Karzinom'
    LABELS = [
        'Fibroadenom', 
        'Adenose', 
        'Lymphknoten', 
        'Zyste', 
        'Fettgewebsnekrose', 
        # 'Hyperplasie', 
        'Duktektasie', 
        # 'Papillom', 
        # 'Hamartom', 
        'Radiäre Narbe', 
        'Duktales Karzinom in situ (DCIS)', 
        'Karzinom'
    ]
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
                CropOrPad((256, 256, 32), padding_position='end', padding_mode=0), # WANRING: Padding mode also for LabelMap
                CropOrPad((224, 224, 32), padding_position='random', padding_mode=0), # WANRING: Padding mode also for LabelMap

                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)), 
                # RescaleIntensity((-2, 2), per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)),

                tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                # tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,10), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1)) if random_flip else tio.Lambda(lambda x: x), # WARNING: Padding mask, DON'T FLIP Z when "end"-padding
                # tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
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
        path_csv = self.path_root/f'metadata/split_regex.csv' 
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
        #     eos_token_id = tokenizer.eos_token_id
        #     df_reports = df_reports[df_reports['Findings'].apply(lambda x: (tokenizer(x) == eos_token_id).nonzero().squeeze().item()<511)] # only 96 cases 161

        self.df_reports = df_reports
        # df_reports[df_reports.index.isin(self.df['UID'])]['Findings'].isna().sum()
        self.df = self.df[self.df['UID'].isin(df_reports.index)]

        # df_files = pd.read_csv(self.path_root/f'metadata/files.csv' )
        # df_files = df_files[['UID', 'Post_1', 'Post_2', 'Post_3', 'Post_4']].dropna()
        # self.df = self.df[self.df['UID'].isin(df_files['UID'])]

        self.item_pointers =  self.df.index.tolist()
        # self.item_pointers =  self.df.index.tolist()[:32*10] * 100

    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)
    
    def get_sub(self, dyn_img):
        dyn_img.load()
        sub_img = torch.stack([dyn_img.data[i]-dyn_img.data[0] for i in range(1, dyn_img.shape[0])])
        sub_img = tio.ScalarImage(tensor=sub_img, affine=dyn_img.affine)
        return sub_img


    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        label = item[self.LABEL]       
        uid = item['UID']
        text = self.df_reports.loc[uid]['Findings']
 

        # dyn_img = self.load_img([self.path_root/'data_unilateral'/uid/img_name for img_name in ['Pre.nii.gz', 'Post_1.nii.gz', 'Post_2.nii.gz', 'Post_3.nii.gz', 'Post_4.nii.gz']])
        dyn_img = self.load_img(self.path_root/'data_unilateral'/uid/'Pre.nii.gz' )
        sub_img = self.load_img(self.path_root/'data_unilateral'/uid/'Sub.nii.gz' )
        t2_img = self.load_img(self.path_root/'data_unilateral'/uid/'T2.nii.gz' )

        mask_fg = dyn_img.data[0].sum(0).sum(0)>0

        # Crop to slices with data 
        dyn_img.set_data(dyn_img.data[:,:,:, mask_fg])
        sub_img.set_data(sub_img.data[:,:,:, mask_fg])
        t2_img.set_data(t2_img.data[:,:,:, mask_fg])
        mask_fg = mask_fg[mask_fg]

        img = tio.Image(tensor=torch.cat([
            dyn_img.data[:,:,:, mask_fg],
            sub_img.data[:,:,:, mask_fg],
            t2_img.data[:,:,:, mask_fg],
        ],dim=0), affine=sub_img.affine )
        mask_fg = mask_fg[mask_fg]

        # mask_fg = tio.LabelMap(tensor=torch.ones_like(dyn_img.data, dtype=torch.int32)*mask_fg[None] , affine=dyn_img.affine)
        # sub = tio.Subject(dyn_img=dyn_img, t2_img=t2_img, mask_fg=mask_fg)

        mask_fg = tio.LabelMap(tensor=torch.ones_like(img.data, dtype=torch.int32)*mask_fg[None] , affine=img.affine)
        sub = tio.Subject(img=img, mask_fg=mask_fg)
    
        sub = self.transform(sub)
        
        img = sub['img']
        # img = torch.cat([sub['dyn_img'], sub['sub_img']], dim=0)
        # img = torch.cat([sub['t2_img'], sub['dyn_img'], sub['sub_img']], dim=0)

        # dyn_img = sub['dyn_img']
        # t2_img = sub['t2_img']
        # img = torch.stack([t2_img[0], dyn_img[0], dyn_img[1]-dyn_img[0], dyn_img[2]-dyn_img[0], dyn_img[3]-dyn_img[0], dyn_img[4]-dyn_img[0]], dim=0)

        mask_fg = sub['mask_fg']
        src_key_padding_mask = ~(mask_fg[0].sum(-1).sum(-1)>0)
        src_key_padding_mask = src_key_padding_mask.repeat(img.shape[0])

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
    
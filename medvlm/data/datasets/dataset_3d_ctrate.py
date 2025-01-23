from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, CropOrPad

from .s3_utils import init_bucket, load_bytesio, load_torchio
import numpy as np 
import ast 

class CTRATE_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/hpcwork/p0020933/workspace_gustav/datasets/CT-CLIP/dataset')
    PATH_ROOT_S3 = Path('CT-RATE/')
    LABELS = [
        'Medical material',
    	'Arterial wall calcification',	
        'Cardiomegaly',	
        'Pericardial effusion',	
        'Coronary artery wall calcification',	
        'Hiatal hernia',	
        'Lymphadenopathy',	
        'Emphysema',	
        'Atelectasis',	
        'Lung nodule',	
        'Lung opacity',	
        'Pulmonary fibrotic sequela',	
        'Pleural effusion',	
        'Mosaic attenuation pattern',	
        'Peribronchial thickening',	
        'Consolidation',	
        'Bronchiectasis',	
        'Interlobular septal thickening'
    ]
    LABEL = 'Lung nodule'
    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            clamp=(-1000, 1000), 
            image_resize = None,# (224,224,96),
            resample=None, # Paper (0.75, 0.75, 1.5)
            image_crop = (420, 308, 210), # Paper: 480 × 480 × 240, NOTE: Must be a multiplier of 14 for Dino
            random_flip = False,
            random_rotate=False,
            random_center=False,
            random_inverse=False,
            random_noise=False, 
            to_tensor = True,
            use_s3=True,
            return_labels=False,
            tokenizer=None,
        ):
        self.path_root = Path((self.PATH_ROOT_S3 if use_s3 else self.PATH_ROOT)if path_root is None else path_root)
        self.path_root_prep = self.path_root/'preprocessed_resample'
        self.tokenizer = tokenizer 
        self.return_labels = return_labels

        if transform is None: 
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                #tio.Lambda(lambda x: x.moveaxis(1, 2)), # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, mask_name='mask', padding_mode=-1000) if image_crop is not None else tio.Lambda(lambda x: x), # WARNING: Padding value also used for mask 

                tio.Clamp(*clamp),
                RescaleIntensity((-1,1), in_min_max=clamp, per_channel=True),

                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                # tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                # tio.RandomFlip((0,1,2)) if random_flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                # tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
                # tio.RandomNoise(std=(0, 0.1)) if random_noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x) 
            ])
        else:
            self.transform = transform

        # Init S3 
        self.use_s3 = use_s3
        if use_s3 and (not hasattr(self, "bucket")):
            self.bucket = init_bucket() 

        # Get split file 
        path_csv = self.path_root/'preprocessed/splits/split.csv'
        if use_s3:
            path_csv = load_bytesio(self.bucket, str(path_csv)) 
        df = self.load_split(path_csv, fold=fold, split=split, fraction=fraction)
        
        # Get reports 
        df_reports = []
        for report_split in ['train', 'validation']:
            path_csv = self.path_root/f'download/radiology_text_reports/{report_split}_reports.csv'
            if use_s3:
                path_csv = load_bytesio(self.bucket, str(path_csv)) 
            df_reports.append(pd.read_csv(path_csv))
        df_reports = pd.concat(df_reports).set_index('VolumeName')
        self.df_reports = df_reports

        # df_reports = df_reports[~df_reports['Impressions_EN'].isna()] # 28 cases 
        # df_reports[~(df_reports['Findings_EN'].isna() & df_reports['Impressions_EN'].isna()) ] # = 0 cases 
        df_reports['Report'] = df_reports['Findings_EN'].fillna("")+df_reports['Impressions_EN'].fillna("")
        # df_reports = df_reports[df_reports['Report'].apply(lambda x: (tokenizer(x)[0].nonzero().max()<511).item())] # ~2000

        # Merge
        df = pd.merge(df, df_reports, how='inner', on='VolumeName')
        self.df = df.set_index('ExamUID', drop=True)
        self.item_pointers = self.df.index.tolist()

        # Remove S3 
        if use_s3:
            del self.bucket

    
    def __len__(self):
        return len(self.item_pointers)
    
    def load_img(self, path_img):
        return load_torchio(self.bucket, path_img) if self.use_s3 else tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return load_torchio(self.bucket, path_img, tio.LABEL) if self.use_s3 else tio.LabelMap(path_img)

    def __getitem__(self, index):
        examuid = self.item_pointers[index] # Note: one exam can have multiple reconstructions 
        item = self.df.loc[examuid]
        folder, patientid, examid = examuid.split('_')
        subfolder = f'{folder}_{patientid}'
        uid = f'{examuid}_{1}' # Always picks the first scan: choose between 1 and 'NumberReconstructions'

        if self.use_s3:
            if not hasattr(self, "bucket"):
                self.bucket = init_bucket() 

        path_item = self.path_root_prep/'data'/folder/subfolder/examuid
        img = self.load_img(path_item/f'{uid}.nii.gz')
        seg_lung = self.load_map(path_item/f'seg_lung.nii.gz')

        # Get Target     
        subject = tio.Subject(img=img, mask=seg_lung)
        subject = self.transform(subject)

        img = subject['img']
        mask = subject['mask']
        mask[mask<0]=0 # workaround padding -1000
        src_key_padding_mask = ~(mask[0].sum(-1).sum(-1)>0)
        # assert ~src_key_padding_mask.all(), "All tokens have been marked as padding tokens"
        label = item[self.LABEL]
        text = item['Report']

        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return {'uid':uid, 'img':img , 'text':text, 'label':label, 'src_key_padding_mask':src_key_padding_mask}
    
    def get_examuid(self, examuid):
        try:
            idx = self.item_pointers.index(examuid)
        except:
            raise ValueError("Index not found")
        return self.__getitem__(idx)
    

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
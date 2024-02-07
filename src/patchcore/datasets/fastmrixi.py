import os
from enum import Enum
import pandas as pd
import PIL
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# Class fastmrixi: uses all FastMRI + IXI images available (all anomalies). 
# Other classes: Used to evaluate model separately on anomalies. Run load_and_evaluate.py for this (run_patchcore.py will train a separate model on each class)
_CLASSNAMES = [
    'fastmrixi', 
    'absent_septum',
    'artefacts',
    'craniatomy',
    'dural',
    'ea_mass',
    'edema',
    'encephalomalacia',
    'enlarged_ventricles',
    'intraventricular',
    'lesions',
    'mass',
    'posttreatment',
    'resection',
    'sinus',
    'wml',
    'other',
    'good'
]

# Note: The backbone is still pre-trained on ImageNet, so we retain this.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val" # Obsolete.
    TEST = "test"

# Dataset class capable of handling both TRAIN and TEST splits.
    
class FastmriIxiDataset(Dataset):
    """
    PyTorch Dataset for FastMRI and IXI images (brain MRI)
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0, # Obsolete
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the FastmriIxi data folder.
            classname: [str or None]. Name of class (fastmrixi, fastmri, or ixi) to use. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to. 
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else ['fastmrixi'] # Default: all anomalies jointly. 
        self.train_val_split = train_val_split # Obsolete
        self.resize = resize
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

        self.pathologies = [
          'absent_septum',
          'artefacts',
          'craniatomy',
          'dural',
          'ea_mass',
          'edema',
          'encephalomalacia',
          'enlarged_ventricles',
          'intraventricular',
          'lesions',
          'mass',
          'posttreatment',
          'resection',
          'sinus',
          'wml',
          'other'
        ]
        self.imagesize = (3, imagesize, imagesize) # 3 channels despite grayscale as pre-trained backbone expects 
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            lambda img: img.convert('RGB'),
            lambda img: transforms.Pad(((img.height - img.width) // 2, 0), fill=0)(img),
            lambda img: img.resize((self.resize,self.resize), PIL.Image.BICUBIC),
            transforms.CenterCrop(self.imagesize[-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            # lambda img: img.convert('L'),
            lambda img: img.resize((self.resize,self.resize), PIL.Image.NEAREST),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
        

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path, _ = self.data_to_iterate[idx] # Last: neg mask path
        image = PIL.Image.open(image_path)
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]]) # Create empty mask for good images. (1,image.size, image.size)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-1]), # TODO check if you can define images uniquely like this.. Rewrite this to match with new image names. As subdir structure of FastMRI and IXI different, this is not sufficient. 
            "image_path": image_path,
        } 

    def __len__(self):
        return len(self.data_to_iterate)

    # Reads in paths, creates splits
    # returns: 
    # ** data_to_iterate: list of [classname, anomaly, img_path, pos_mask_path, neg_mask_path]
    # ** imgpaths_per_class: dict of [classname][anomaly] : list of paths
    def get_image_data(self):
        split_dir = os.path.join(self.source, "splits")

        imgpaths_per_class = {}
        for classname in self.classnames_to_use:

          imgpaths_per_class[classname] = {}

          if self.split == DatasetSplit.TRAIN:
            train_csv_ixi = os.path.join(split_dir, 'ixi_normal_train.csv')
            train_csv_fastMRI = os.path.join(split_dir, 'normal_train.csv')
            val_csv = os.path.join(split_dir, 'normal_val.csv')

            train_files_ixi = pd.read_csv(train_csv_ixi)['filename'].tolist()
            train_files_fastMRI = pd.read_csv(train_csv_fastMRI)['filename'].tolist()
            val_files_fastMRI = pd.read_csv(val_csv)['filename'].tolist()

            # Combine files
            train_data = train_files_ixi + train_files_fastMRI # Note: to have comparable results with group mate, val set is not added (although in theory we could extend training data with val here, as PatchCore does not need val set) 
            train_data = [os.path.join(self.source,path.split("./data/")[1]) for path in train_data] # In csvs: data is under ./data. Here: ./data/fastmrixi. #TODO check if this still holds with new classes
            
            # We only have anomaly="good" in training set
            imgpaths_per_class[classname]["good"] = train_data
            data_to_iterate = [[classname, "good", img_path, None, None] for img_path in train_data] 

          elif self.split == DatasetSplit.TEST:
            if classname == "fastmrixi":
                ### First, we add some normal images to the test set. 
                normal_csv = os.path.join(split_dir, 'normal_test.csv')
                normal_paths = pd.read_csv(normal_csv)['filename'].tolist()
                normal_paths = [os.path.join(self.source,path.split("./data/")[1]) for path in normal_paths] # In csvs: data is under ./data. Here: ./data/fastmrixi.
                imgpaths_per_class[classname]["good"] = [normal_paths]
                # Read in negative mask paths (or "foreground" of test data). This will be later used to calculate pixelwise metrics.
                neg_mask_csv = os.path.join(split_dir, "normal_test_ann.csv")
                neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist()
                neg_mask_paths = [os.path.join(self.source,path.split("./data/")[1]) for path in neg_mask_paths] # In csvs: data is under ./data. Here: ./data/fastmrixi.

                data_to_iterate = [[classname, "good", img_path, None, neg_mask_path] for img_path, neg_mask_path in zip(normal_paths, neg_mask_paths)] 
                assert(len(normal_paths) == len(neg_mask_paths))

                #### Then we add anomalous images. 
                for pathology in self.pathologies:
                    imgpaths_per_class[classname][pathology] = []

                    img_csv = os.path.join(split_dir, f'{pathology}.csv')
                    pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
                    neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv') 

                    img_paths = pd.read_csv(img_csv)['filename'].tolist()
                    pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
                    neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist() 
                    # In csvs: data is under ./data. Here: ./data/fastmrixi. 
                    img_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in img_paths]
                    pos_mask_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in pos_mask_paths]
                    neg_mask_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in neg_mask_paths] 
                    assert len(img_paths) == len(pos_mask_paths) == len(neg_mask_paths) 

                    imgpaths_per_class[classname][pathology].append(img_paths)
                    data_to_iterate = data_to_iterate + [[classname, pathology, img_path, mask_path, neg_mask_path] for img_path, mask_path, neg_mask_path in zip(img_paths, pos_mask_paths, neg_mask_paths)] # TODO extension wih neg mask
            
            elif classname == "good":
               ### First, we add some normal images to the test set. 
                normal_csv = os.path.join(split_dir, 'normal_test.csv')
                normal_paths = pd.read_csv(normal_csv)['filename'].tolist()
                normal_paths = [os.path.join(self.source,path.split("./data/")[1]) for path in normal_paths] # In csvs: data is under ./data. Here: ./data/fastmrixi.
                imgpaths_per_class[classname]["good"] = [normal_paths]
                # Read in negative mask paths (or "foreground" of test data). This will be later used to calculate pixelwise metrics.
                neg_mask_csv = os.path.join(split_dir, "normal_test_ann.csv")
                neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist()
                neg_mask_paths = [os.path.join(self.source,path.split("./data/")[1]) for path in neg_mask_paths] # In csvs: data is under ./data. Here: ./data/fastmrixi.

                data_to_iterate = [[classname, "good", img_path, None, neg_mask_path] for img_path, neg_mask_path in zip(normal_paths, neg_mask_paths)] 
                assert(len(normal_paths) == len(neg_mask_paths))
            
            else: # Class is an anomaly
                pathology = classname
                imgpaths_per_class[classname][pathology] = []
                data_to_iterate = []

                img_csv = os.path.join(split_dir, f'{pathology}.csv')
                pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
                neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv') 

                img_paths = pd.read_csv(img_csv)['filename'].tolist()
                pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
                neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist() 
                # In csvs: data is under ./data. Here: ./data/fastmrixi. 
                img_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in img_paths]
                pos_mask_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in pos_mask_paths]
                neg_mask_paths = [os.path.join(self.source, path.split("./data/")[1]) for path in neg_mask_paths] 
                assert len(img_paths) == len(pos_mask_paths) == len(neg_mask_paths) 

                imgpaths_per_class[classname][pathology].append(img_paths)
                data_to_iterate = data_to_iterate + [[classname, pathology, img_path, mask_path, neg_mask_path] for img_path, mask_path, neg_mask_path in zip(img_paths, pos_mask_paths, neg_mask_paths)] # TODO extension wih neg mask
               
          else: # Validation split
            print("VAL split is not needed for PatchCore and not implemented! Did you really mean to use this?")
            return None, None

        return imgpaths_per_class, data_to_iterate
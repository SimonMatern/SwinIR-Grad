import torch

from torchvision.io import read_image
import glob
from models.network_swinir import SwinIR3D
import yaml
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import argparse
import torch.nn.functional as F
from train import SwinIR_PL

class TestSet(Dataset):
    def __init__(self, data_dirs, lr_size=None, hr_size=None, len=1, augm=None):
        self.data_dirs = data_dirs
        self.lr_size = lr_size
        self.hr_size = hr_size

        if (lr_size is not None) and (hr_size is not None):
            self.lr_size = Resize(lr_size)
            self.hr_size = Resize(hr_size)
        self.imgs = sum([glob.glob(directory +"/**/*.*", recursive=True) for directory in data_dirs],[])
        
        if augm=="color":
            augs =[
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.WithBrightnessChannels(iaa.Add((-100, 100))),
                iaa.BlendAlphaSimplexNoise(
                foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
            )]
            def augment(images,):
                aug = lambda x: np.random.choice(augs)(images=[x])[0]
                res = [aug(img) for img in images]
                return res
            
            self.augm = augment
        elif augm=="blur":
            augs_blur =[
                iaa.imgcorruptlike.Pixelate(severity=(1,5)),
                iaa.imgcorruptlike.MotionBlur(severity=(1,5)),
                iaa.imgcorruptlike.DefocusBlur(severity=(1,5))]

            def augment(images,):
                aug = lambda x: np.random.choice(augs_blur)(images=[x])[0]
                res = [aug(img) for img in images]
                return res
            self.augm = augment
        elif augm=="jpeg":
            self.augm = iaa.JpegCompression((50,99))
        else:
            self.augm = augm
        self.len = len

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = read_image(img_path)

        image_lr = image
        image_hr = image 
        if self.lr_size is not None:
            image_lr = self.lr_size(image)
            image_hr = self.hr_size(image)

        if self.augm is None:
            return image_lr/255., image_hr/255.
        
        else:
            image_lr = image_lr.permute(1, 2, 0).numpy()
            images_lr = [self.augm(images=[image_lr])[0] for _ in range(self.len)]
            images_lr = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in images_lr])

            window_size = 4
            _, _, h_old, w_old = images_lr.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            h_pad = h_pad//2, h_pad - h_pad//2
            w_pad = (w_old // window_size + 1) * window_size - w_old
            w_pad = w_pad//2, w_pad - w_pad//2
            images_lr = F.pad(images_lr, (*w_pad,*h_pad), "constant", 0)
            image_hr = F.pad(image_hr, (*w_pad,*h_pad), "constant", 0)

            return images_lr/255.0,image_hr/255.0

    
def model_from_ckpt(path, device="cpu"):

    with open(path+ "/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    ckpt_path = glob.glob(path+'/**/*.ckpt', recursive=True)
    if len(ckpt_path)>1:
        print(f"multiple checkpoints exist. \n {ckpt_path[0]} is used")
    ckpt_path = ckpt_path[0]
    checkpoint = torch.load(ckpt_path,  map_location=torch.device(device))
    del config["model"]['grad_weight']
    model = SwinIR_PL(**config["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)


    return model, config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    pass
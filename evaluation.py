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

from einops import rearrange
import torch 
import glob
import cv2 as cv
import numpy as np
from utils import util_calculate_psnr_ssim as util
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import os
import random
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
    model = SwinIR_PL(**config["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)


    return model, config





def evaluation(ckpt_path, data_dirs, N=8, samples=1, device="cpu"):

    net,config = model_from_ckpt(ckpt_path, device=device)
    name = config["trainer"]["name"]

    augm = config["data"]["augm"]
    
    print(f"Evalutation of {name} on task ’{augm}’ with N={N} \n Checkpoint:{ckpt_path}")

    data = TestSet(data_dirs=data_dirs, augm=augm, len=N)

    border = 0

    test_results = OrderedDict()
    test_results['file'] = []
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            for _ in range(samples):
                x,y = data[i]

                file = data.imgs[i]
                x = x.to(device)

                y = y.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if y.ndim == 3:
                    y = np.transpose(y[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = net(x.unsqueeze(0))
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                # save image
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR

                output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

                img_gt = (y * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = np.squeeze(img_gt)

                psnr = util.calculate_psnr(output, img_gt, crop_border=border)
                ssim = util.calculate_ssim(output, img_gt, crop_border=border)


                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                psnrb = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=False)
                psnrb_y = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)

                test_results["file"].append(file)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)   
                test_results['psnrb'].append(psnrb)
                test_results['psnrb_y'].append(psnrb_y)   
                #print(f'Testing {i:d} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}; PSNRB: {psnrb:.2f} dB;'
                #        f'PSNR_Y: {psnr_y:.2f} dB; SSIM_Y: {ssim_y:.4f}; PSNRB_Y: {psnrb_y:.2f} dB.')
                #cv.imwrite(f'{ckpt_path}/{i}_{name}.png', output)
                    # summarize psnr/ssim

    df = pd.DataFrame.from_dict(test_results)
    df.to_csv(f'{ckpt_path}/{N}.csv', index=False) 

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(name, ave_psnr, ave_ssim))
    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
    print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
    ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
    print('-- Average PSNRB: {:.2f} dB'.format(ave_psnrb))
    ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
    print('-- Average PSNRB_Y: {:.2f} dB'.format(ave_psnrb_y))


if __name__ == '__main__':
    
    data_dirs = ["/shared/DIV2K/TestSet/BSDS100/","/shared/DIV2K/TestSet/Set14/", "/shared/DIV2K/TestSet/Set5/", "/shared/DIV2K/TestSet/urban100/", "/shared/DIV2K/TestSet/manga109//"]
    ckpts = list(reversed(sorted(glob.glob("lightning_logs/version*"))))
    random.shuffle(ckpts)
    print(ckpts)
    for N in list(reversed([2, 4, 8, 10, 16, 20, 30, 40, 50])):
        for ckpt_path in ckpts:

                torch.cuda.empty_cache()
                # if os.path.isfile(f'{ckpt_path}/{N}.csv'):
                #     print("Evaluation already exists")
                #     continue
                try:
                    evaluation(ckpt_path, data_dirs, N=N,  samples=1, device="cuda:0")
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, retrying on CPU')
                        evaluation(ckpt_path, data_dirs, N=N,  samples=1, device="cpu")

                except Exception as e:
                    print('An exception occurred in {}: {}'.format(ckpt_path,e))
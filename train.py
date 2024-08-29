import torch 
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import glob
from models.network_swinir import SwinIR
from gradients import Tensor2Gradient
import lightning as L
from lightning.pytorch.cli import LightningCLI
from kornia.losses import ssim_loss, psnr_loss
#from lightning.pytorch.loggers import TensorBoardLogger
from models.model import DeepAggNet
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import numpy as np

class DIV2K(Dataset):
    def __init__(self, img_dir, lr_size=(256,256), hr_size=(256,256), len=1, augm=None):
        self.img_dir = img_dir
        self.lr_size = Resize(lr_size)
        self.hr_size = Resize(hr_size)
        self.files = sorted(glob.glob(img_dir+"/*"))
        
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
        else:
            self.augm = augm
        self.len = len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = read_image(img_path)

        image_lr = self.lr_size(image)
        image_hr = self.hr_size(image)

        if self.augm is None:
            return image_lr/255., image_hr/255.
        
        else:
            image_lr = image_lr.permute(1, 2, 0).numpy()
            images_lr = [self.augm(images=[image_lr])[0] for _ in range(self.len)]
            images_lr = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in images_lr])

            return images_lr/255.0,image_hr/255.0

    


class SwinIR_PL(L.LightningModule):
    """ SwinIR as PL-Module"""
    def __init__(self, grad_weight=4.0, **kwargs):
        """
        """
        super(SwinIR_PL, self).__init__()

        #self.model = SwinIR(**kwargs)
        self.model = DeepAggNet(**kwargs)
        self.loss = nn.L1Loss()
        self.gradient =  Tensor2Gradient()
        self.w = grad_weight
        self.lr = 1e-3

    def forward(self, x):

        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        x,y = batch
        
        y_grad = self.gradient(y)

        if self.model.use_gradients:
            pred_rgb,pred_grad  = self.model(x)
        else:
            pred_rgb  = self.model(x)
            pred_grad = self.gradient(pred_rgb)

        loss_grad = self.loss(y_grad,pred_grad)
        loss_rgb = self.loss(y,pred_rgb)
        loss = loss_rgb + self.w*loss_grad

        ssim = ssim_loss(pred_rgb,y,7)
        psnr = psnr_loss(pred_rgb,y,1.0)

        self.log("L_rgb",loss_rgb, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_grad",loss_grad, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_ssim",ssim, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_psnr",psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x,y = batch
        y_grad = self.gradient(y)

        if self.model.use_gradients:
            pred_rgb, pred_grad  = self.model(x)
        else:
            pred_rgb  = self.model(x)
            pred_grad = self.gradient(pred_rgb)

        loss_grad = self.loss(y_grad,pred_grad)
        loss_rgb = self.loss(y,pred_rgb)
        loss = loss_rgb + self.w*loss_grad

        ssim = ssim_loss(pred_rgb,y,7)
        psnr = psnr_loss(pred_rgb,y,1.0)

        self.log("L_rgb_val",loss_rgb, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_grad_val",loss_grad, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_ssim_val",ssim, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_psnr_val",psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L_loss_val",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=(self.lr or self.learning_rate) )
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        #return [optimizer], [lr_scheduler]
        return {
           'optimizer': optimizer,
           #'lr_scheduler': lr_scheduler, # Changed scheduler to lr_scheduler ,'monitor': 'val_loss'
       }

class DIV2K_PL(L.LightningDataModule):
    
    def __init__(self, train, val, lr_size=(256,256), hr_size=(512,512), batch_size=32, augm=None, len=1, **kwargs):
        super(DIV2K_PL, self).__init__()
        self.train = train
        self.val = val
        self.batch_size = batch_size
        self.lr_size = lr_size
        self.hr_size = hr_size

        self.augm = augm
        self.len=len
        self.prepare_data()
    
    def prepare_data(self):
        self.train_data = DIV2K(self.train, self.lr_size, self.hr_size, self.len, self.augm) 
        self.val_data = DIV2K(self.val, self.lr_size, self.hr_size, self.len, self.augm) 

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass



    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=16 )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=16)

def trainer(name,**args)-> L.Trainer: 
    #logger = TensorBoardLogger(save_dir="tb_logs", name=name)
    trainer = L.Trainer(**args)
    return trainer


def cli():
    LightningCLI(SwinIR_PL,DIV2K_PL,trainer_class=trainer, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli()

# python train.py -c config config.yaml
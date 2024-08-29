from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchvision
from models.swintransformer import BasicLayer, PatchEmbed3D
from einops import rearrange
from kornia.losses import ssim_loss

from gradients import Tensor2Gradient, Gradient2Tensor
toPIL = transforms.ToPILImage()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def linear_stretch(x):
    x_min = x.amin((2,3), True)
    x_max = x.amax((2,3), True)
    return (x-x_min)/(x_max-x_min)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
    
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, dilation=dilation,padding=dilation, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
     
class MeanAgg(nn.Module):
    def __init__(self):
        super(MeanAgg, self).__init__()

    def forward(self, x):
        """ Forward-pass of mean-aggregation
        x: list of tensor of size (B, S, C, H, W)
        """
        return x.mean(1)    
        

class OffsetEstimation(nn.Module):
    def __init__(self, size=(8,8), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.linear = nn.Linear(size[0]*size[1]*3,3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=size[0]*size[1]*3, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        
        B, N, C, H, W = x.size()
        x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.pooling(x)
        x = rearrange(x, "(b n) c h w -> b n (c h w)", b=B)
        x = self.transformer_encoder(x).mean(1)
        x = self.linear(x).unsqueeze(-1).unsqueeze(-1)

        return x
    
class SwinTransAgg(nn.Module):
    """ This Block uses Video Swin Transformers to process and aggregate a series of 2D feautures"""
    def __init__(self, in_dim = 64, depth=4, heads=8, patch_size=(2,4,4), window_size = (1,7,7), mode= "softmax"):
        super(SwinTransAgg, self).__init__()

        self.softmax = nn.Softmax(2)
        self.mode = mode
        if patch_size[1] != patch_size[2]:
            raise Exception("Only square patches allowed")
        P = patch_size[1]
        self.tokenize = PatchEmbed3D(patch_size=patch_size, in_chans=in_dim, embed_dim=P*P*in_dim)
        self.shuffle = nn.PixelShuffle(P)
        self.transformer_encoder = BasicLayer(dim=P*P*in_dim, depth=depth, num_heads=heads, window_size = window_size) if (mode != "deepset") else nn.Identity()



    def forward(self, xs):
        """Forward pass of our Aggregation Block 
        
        xs: list of tensors of size (B, S, C, H, W)
            S: Sequence Length
            B: batch size
            C: Channels (= embed_dim)
            H: Height
            W: Width 
        """
        B, S, C, H, W = xs.size()                   # -> (B, S, C, H, W)
        if self.mode == "deepset":
            return xs.mean(1)    


        xs = rearrange(xs, 'b s c h w -> b c s h w')
        xs = self.tokenize(xs)                        # -> (B, P*P*C, S/patch_size[0], H/P, W/P)
        y = self.transformer_encoder(xs)             # -> (B, P*P*C, S/patch_size[0], H/P, W/P)
        #y = rearrange(y, 'b c s h w -> b s c h w')

        if self.mode == "token":
            return y[:,0,:,:]
        if self.mode == "softmax":
            y = self.softmax(y)                     # -> (B, C, S, H, W) normalized along (S)equence dimension
            y = (y*xs).sum(2)                       # -> (B, C, H, W)
        if self.mode == "softmax2":
            y = self.softmax(y)                     # -> (B, S, C, H, W) normalized along (S)equence dimension
            y = y.sum(2)                            # -> (B, C, H, W)
        if self.mode == "sum":
            y = y.sum(2)    
        if self.mode =="mean":
            y = y.mean(2)    
        if self.mode =="median":
            y = torch.median(y,2).values

        y = self.shuffle(y)    # (B, C*P*P, H/P, W/P) -> (B, C, H, W)
        return y
    
    
class DeepAggNet(nn.Module):
    """ Deep AGG Neural Network with Transformers"""
    def __init__(self, encoder_num_blocks=10, decoder_num_blocks=10, smooth_num_blocks=6, 
                 planes=32, downsampling_factor=2, use_gradients=True, mixed=False,
                 swin_depth=4,swin_num_heads=8, window_size = (1,7,7), patch_size=(2,4,4),
                 mode="softmax", **kwargs):
        """
        encoder_num_blocks: Number of residual blocks used for encoding the images into an embedding
        decoder_num_blocks: Number of residual blocks used for decoding the embeddings into an image
        smooth_num_blocks:  Number of residual blocks used for smoothing the upsampled/decoded embedding
        planes:             Number of feature planes used in the initial embedding,
                            the number of planes double after each downsampling
        agg_block:          A block that aggregates a series of embeddings into a singular embedding: 
                            (B, S, C, H, W) -> (B, C, H, W)
        """
        super(DeepAggNet, self).__init__()
        
        self.mode = mode
        self.use_gradients= use_gradients
        self.mixed = mixed

        self.planes = planes

        k = 9 if (self.mixed) else (6 if self.use_gradients else 3) 
        self.input = nn.Conv2d(k, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.output= nn.Conv2d(self.planes, k, kernel_size=3, stride=1, padding=1, bias=True)

        if self.mixed:
            self.mix_layer = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=True)
        # gradient transformation
        embedding_size = 256// (2**downsampling_factor)
        gradient = Tensor2Gradient()
        gradient_inv = Gradient2Tensor(embedding_size,embedding_size)
        self.Tensor2Gradient = gradient if use_gradients else nn.Identity()
        self.Gradient2Tensor = gradient_inv  if use_gradients else nn.Identity()
        

        # Create a down-/up-sampling architecture
        self.downsample = []
        self.upsample = []
        n = planes
        for i in range(downsampling_factor):
            # create downsampling layers using convolutions with strides
            self.downsample.append( nn.Conv2d(in_channels = n, out_channels=n*2, kernel_size=3, stride=2, padding=1 ) )
            self.downsample.append(nn.ReLU(inplace=True))

            # create upsampling layers using transposed convolutions (should be symmetric to downsampling)
            self.upsample = [nn.ReLU(inplace=True)] + self.upsample
            self.upsample = [nn.ConvTranspose2d(in_channels=n*2, out_channels=n, kernel_size=3, stride=2, padding=1, output_padding=1)] + self.upsample
            n *= 2
            
        
        self.downsample = nn.Sequential(*self.downsample)
        self.upsample = nn.Sequential(*self.upsample)
        
        
        # Embedding of downsampled features
        self.encoder = self._make_layer(n, encoder_num_blocks)
        
                                        

        self.agg = SwinTransAgg(in_dim=n, depth=swin_depth, heads = swin_num_heads, window_size=window_size,patch_size=patch_size, mode=mode, **kwargs)
         

        # create decoder layers that are applied on the aggregated features
        self.decoder = self._make_layer(n, decoder_num_blocks)
        
        
        # create smoothing layers that are applied on the upsampled features
        self.smooth  = self._make_smooth_layer(planes, smooth_num_blocks)
        
        

        
    def _make_layer(self, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(DilatedResidualBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    
    def _make_smooth_layer(self, planes, num_blocks):
        layers = []
        dilation = 1
        for i in range(num_blocks):
            layers.append(DilatedResidualBlock(planes,planes,dilation))
            if i%2 == 0:
                dilation *= 2
        layers.append( nn.Conv2d(in_channels = planes, out_channels=planes, kernel_size=3, stride=1, padding=1 ) )
        layers.append(nn.ReLU(inplace=True))
        layers.append( nn.Conv2d(in_channels = planes, out_channels=planes, kernel_size=3, stride=1, padding=1 ) )
        return nn.Sequential(*layers)
            
        

    def forward(self, x):
        """Forward pass of our Deep Aggregation Network 
        
        x: of tensor of size (B, S, C, H, W)
        """
        B, S, C, H, W = x.size()

        x = rearrange(x, 'b s c h w -> (b s) c h w')

        #xs = torch.split(x,1,dim = 1)  # (B, S, C, H, W) -> [(B, 1, C, H, W)]
        #xs = [torch.squeeze(x,dim=1) for x in xs] #  [(B, 1, C, H, W) x S] ->  [(B, C, H, W) x S]
        grad = self.Tensor2Gradient(x) # apply gradient

        if self.mixed:
            x = torch.cat((grad,x),1)
        else:
            x = grad
        embedding = self.encoder(self.downsample(self.input(x)))
        
        # Compute the average image intensity for each channel
        #avg = torch.stack(embedding,1).mean(1).mean([2,3], True) #  [(B, C, H, W) x S] -> (B, C, 1, 1)
        
        #embedding = torch.stack(embedding,1) #  [(B, C, H, W) x S] -> (B, S, C, H, W)
        embedding = rearrange(embedding, '(b s) c h w -> b s c h w', b=B,s=S)
        embedding = self.agg(embedding) # (B, S, C, H, W) -> (B, C, H, W)
        
        out = self.output(self.smooth(self.upsample(self.decoder(embedding))))

        if self.mixed:

            #split channels
            grad, rgb = torch.split(out, (6,3), 1)

            # transform gradients back to rgb
            grad2rgb = self.Gradient2Tensor(grad)

            # fuse both channels
            out = torch.cat((grad2rgb,rgb),1)
            out = self.mix_layer(out), grad
        return out


                        


class DeepAggNetPL(pl.LightningModule):
    """ Deep AGG Neural Network with Transformers"""
    def __init__(self, data_dir, batch_size=32, lr=1e-3, N=10, use_gradient_domain=False, offset_estimation=False, **kwargs):
        """
        encoder_num_blocks: Number of residual blocks used for encoding the images into an embedding
        decoder_num_blocks: Number of residual blocks used for decoding the embeddings into an image
        smooth_num_blocks:  Number of residual blocks used for smoothing the upsampled/decoded embedding
        planes:             Number of feature planes used in the initial embedding,
                            the number of planes double after each downsampling
        agg_block:          A block that aggregates a series of embeddings into a singular embedding: 
                            (B, S, C, H, W) -> (B, C, H, W)
        """
        super(DeepAggNetPL, self).__init__()
                
        self.loss = nn.L1Loss()
        
        self.lr=lr
        self.batch_size = batch_size
        self.use_gradient_domain = use_gradient_domain
        self.net = DeepAggNet( **kwargs)
        

        self.save_hyperparameters(kwargs)
        data = sidar.BlenderDataset(data_dir, N)
        n = len(data)

        train, val, test  = (0.8, 0.1, 0.1)
        train = int(train*n)
        val = int(val*n)
        test = n - train - val

        train_data, val_data, test_data = random_split(data, [train,val,test], generator=torch.Generator().manual_seed(45))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data


        # gradient transformation
        embedding_size = 256
        gradient =  Tensor2Gradient()
        gradient_inv = Gradient2Tensor(embedding_size,embedding_size)
        self.Tensor2Gradient = gradient if use_gradient_domain else nn.Identity()
        self.Gradient2Tensor = gradient_inv  if use_gradient_domain else nn.Identity()

        self.gradient = Tensor2Gradient()
        if offset_estimation:
            self.offset = OffsetEstimation((2,2))
        else:
            self.offset = lambda x: x.mean(1).mean([2,3], True)

    def forward(self, x):
        """Forward pass of our DeepSet Network 
        
        x: of tensor of size (B, S, C, H, W)
        """

        # Compute the average image intensity for each channel
        avg = self.offset(x) #  [(B, C, H, W) x S] -> (B, C, 1, 1)
        out = self.net(x) 


        out = self.Gradient2Tensor(out) 
        if self.use_gradient_domain:
            out += avg
        return out
    
    def training_step(self, batch, batch_idx):
        """Forward pass of our DeepSet Network 
        
        batch : tuple of tensors of size (B, S, C, H, W)
        """
        # training_step defined the train loop. It is independent of forward
        x, y = batch



        # Compute the average image intensity for each channel
        #avg = x.mean(1).mean([2,3], True) #  [(B, C, H, W) x S] -> (B, C, 1, 1)
        avg = self.offset(x)
        # Forward pass
        out = self.net(x) 

        # Transform gradient to RGB
        # This function is an identity mapping if use_gradient_domain=False
        out_rgb = self.Gradient2Tensor(out) 



        if self.use_gradient_domain:
            y_grad = self.Tensor2Gradient(y) #  transform label to gradient domain
            out_rgb += avg
            loss_grad = self.loss(y_grad, out)
            self.log('train_loss_grad', loss_grad)
        else:
            y_grad = self.gradient(y) #  transform label to gradient domain
            out_grad = self.gradient(out_rgb) #  transform output to gradient domain
            loss_grad = self.loss(y_grad, out_grad)
            self.log('train_loss_grad', loss_grad)


        loss = self.loss(y, out_rgb)

        ssim = ssim_loss(y,out_rgb,7)
        #self.logger.experiment.log_metric({'train_loss':loss.item()})


        #self.logger.experiment.log_image(seqs[0])
        #logs = {'train_loss':loss}
        self.log('train_ssim', ssim)
        self.log('train_loss', loss)


        w = 4.0
        loss = loss + w*loss_grad
        return {'loss': loss}
    
    
    def validation_step(self, batch, batch_idx):
        """Forward pass of our DeepSet Network 
        
        batch : tuple of tensors of size (B, S, C, H, W)
        """
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        #x = self.zeropad(x)
        #y = self.zeropad(y)
  

        # Compute the average image intensity for each channel
        #avg = x.mean(1).mean([2,3], True) #  [(B, C, H, W) x S] -> (B, C, 1, 1)
        avg = self.offset(x)
        # Forward pass
        out = self.net(x) 

        out_rgb = self.Gradient2Tensor(out) 


        if self.use_gradient_domain:
            y_grad = self.Tensor2Gradient(y) #  transform label to gradient domain
            out_rgb += avg
            loss_grad = self.loss(y_grad, out)
            self.log('val_loss_grad', loss_grad)
        else:
            y_grad = self.gradient(y) #  transform label to gradient domain
            out_grad = self.gradient(out_rgb) #  transform output to gradient domain
            loss_grad = self.loss(y_grad, out_grad)
            self.log('val_loss_grad', loss_grad)

        loss = self.loss(y, out_rgb)    
        ssim = ssim_loss(y,out_rgb,7)
        
        self.log('val_ssim', ssim)
        self.log('val_loss', loss)


        if batch_idx % 50 == 0:
            grid_out = torchvision.utils.make_grid(out_rgb) 

            self.logger.experiment.log_image(toPIL(grid_out), name="reconstructions"+str(batch_idx) , step =self.global_step) 
            if self.current_epoch <1:
                grid_y = torchvision.utils.make_grid(y) 
                self.logger.experiment.log_image(toPIL(grid_y), name="label"+str(batch_idx) , step =self.global_step) 
        
        return {'loss_val': loss, "input": x, "gt":y, "output": out}
    
#    def validation_epoch_end(self, outputs):
#        x = outputs[0]
#        grid = torchvision.utils.make_grid(x["output"]) 
#        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#        try:
#            self.logger.experiment.log_image(toPIL(grid), name="reconstructions" , step =self.global_step) 
#        except: 
#            pass
#        return {'avg_loss': avg_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=(self.lr or self.learning_rate) )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        #return [optimizer], [lr_scheduler]
        return {
           'optimizer': optimizer,
           'lr_scheduler': lr_scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }
        #return [optimizer], [lr_scheduler]
        
    def train_dataloader(self):
        return DataLoader(self.train_data,pin_memory=True,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data,pin_memory=True,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,pin_memory=True,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)
        


import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision
import numpy as np

class RGB2Mixed(nn.Module):
    """Converts a batch of images into rgb + gradient

    """
    def __init__(self, mix = True):
        super(RGB2Mixed, self).__init__()
        self.Tensor2Gradient = Tensor2Gradient()
        self.mix = mix

    def forward(self,x):
        #split channels
        grad = self.Tensor2Gradient(x) 

        if self.mix:
            # fuse both channels
            out = torch.cat((grad,x),1)
            return out
        return grad

class Mixed2RGB(nn.Module):
    """Converts a batch of images containing rgb + gradient channels into rgb

        mode: ["mix" "mean", "grad","id"]
    """
    def __init__(self, img_size, mode ="mix"):
        super(Mixed2RGB, self).__init__()
        self.Gradient2Tensor = Gradient2Tensor(img_size[0],img_size[1])
        self.mode = mode

        if mode =="mix":
            self.mix_layer = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=True) 

    def forward(self,x, offset=None):
        
        if offset is None:
            offset = 0
        
        if (self.mode == "grad"):
            return self.Gradient2Tensor(x)+offset,x
        elif (self.mode == "id"):
            return x
        
        #split channels
        grad, rgb_out = torch.split(x, (6,3), 1)
        # transform gradients back to rgb
        grad_out = self.Gradient2Tensor(grad)+offset

        if (self.mode == "mix"):
            #fuse both channels
            out = torch.cat((grad_out,rgb_out),1)
            return self.mix_layer(out), grad
        elif (self.mode == "mean"):
            # avg
            return 0.5*(grad_out + rgb_out), grad
        return x
    


class Tensor2Gradient(nn.Module):
    """Converts a batch of images into their x- and y-gradients

    """
    def __init__(self, padding = "circular"):
        super(Tensor2Gradient, self).__init__()
        self.register_buffer("x_grad", 
            torch.tensor([[0,0,0],
                         [-1,1.0,0],
                          [0,0,0]],).unsqueeze(0).unsqueeze(0))
        self.register_buffer("y_grad", 
            torch.tensor([[0,-1,0],
                         [0,1.0,0],
                          [0,0,0]],).unsqueeze(0).unsqueeze(0))

        
    def forward(self,x):
        #x = self.zeropad(x)
        x = F.pad(x, (1,1,1,1), "circular")
        B,C,H,W = x.size()
        x = rearrange(x, 'b c h w -> (b c) 1 h w')
        #x = torch.log(x + 0.001)
        x_d = F.conv2d(x, self.x_grad)
        y_d = F.conv2d(x, self.y_grad)

        x_d = rearrange(x_d, '(b c) 1 h w -> b c h w',c=C)
        y_d = rearrange(y_d, '(b c) 1 h w -> b c h w',c=C)
        
        return torch.cat([x_d,y_d],1)
    

class Gradient2Tensor(nn.Module):
    def __init__(self, H=256, W=256):
        super(Gradient2Tensor, self).__init__()
        
        self.H = H
        self.W = W
    
        inv_laplacian = self.inverse_laplacian(H,W)
        self.register_buffer("inv_laplacian", inv_laplacian.unsqueeze(0).unsqueeze(0))

        self.register_buffer("x_grad_flip", 
            torch.tensor([[0,0,0],
                         [0,1.0,-1],
                          [0,0,0]],).unsqueeze(0).unsqueeze(0))
        
        self.register_buffer("y_grad_flip", 
            torch.tensor([[0,0,0],
                         [0,1.0,0],
                          [0,-1,0]],).unsqueeze(0).unsqueeze(0))
        
    def inverse_laplacian(self, H,W):
        # Laplacian filter with cyclic shift
        laplacian = torch.zeros((H,W))
        laplacian[0,0]=4
        laplacian[0,1]=-1
        laplacian[1,0]=-1
        laplacian[0,-1]=-1
        laplacian[-1,0]=-1

        

        # FFT of laplace filter
        laplacian = torch.fft.rfft2(laplacian)
         # Zero Handling
        T = 1.0e-32
        zero =  torch.abs(laplacian) < T
        laplacian[zero] = 1.0
        laplacian = 1.0/laplacian
        #laplacian[zero] = 0

        return laplacian
    
    def forward(self, x):


        B,C,H,W = x.size()
        x_grad, y_grad = torch.split(x, C//2,1)
        x_grad = rearrange(x_grad, 'b c h w -> (b c) 1 h w')
        y_grad = rearrange(y_grad, 'b c h w -> (b c) 1 h w')

        x_grad = F.pad(x_grad, (1,1,1,1), "circular")
        y_grad = F.pad(y_grad, (1,1,1,1), "circular")


        x_grad = F.conv2d(x_grad, self.x_grad_flip)
        y_grad = F.conv2d(y_grad, self.y_grad_flip)

        x = x_grad+y_grad

        # FFT of gradient
        dtype = x.dtype
        x_fft = torch.fft.rfft2(x.double())
        
        
        if (H ==self.H and W ==self.W):
            inv_laplacian = self.inv_laplacian
        else:
            #print("Computing (Inverse) Laplacian ...")
            inv_laplacian = self.inverse_laplacian(H,W).to(x.device)

        # Division in Frequency Domain
        x_fft = x_fft * inv_laplacian
        #x_fft[:,:,zero] = 0
        x = torch.fft.irfft2(x_fft).to(dtype)

        x = rearrange(x, '(b c) 1 h w -> b c h w', c= C//2)
        return x

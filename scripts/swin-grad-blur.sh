#!/bin/bash

python train.py fit  -c configs/swin-grad-blur.yaml --trainer.devices [0] --trainer.name "SwinIR-RGB" --model.use_gradients false --model.mixed false --model.grad_weight 0.0 

python train.py fit  -c configs/swin-grad-blur.yaml --trainer.devices [0] --trainer.name "SwinIR-RGB++" --model.use_gradients false --model.mixed false --model.grad_weight 4.0

python train.py fit  -c configs/swin-grad-blur.yaml --trainer.devices [0] --trainer.name "SwinIR-Mix" --model.use_gradients true --model.mixed true  --model.grad_weight 4.0 #--model.mode "mix" 


# python train.py fit  -c configs/swin-grad-jpeg.yaml --trainer.devices [0] --trainer.name "SwinIR-RGB" --model.use_gradients false --model.mixed false --model.grad_weight 0.0 

# python train.py fit  -c configs/swin-grad-jpeg.yaml --trainer.devices [0] --trainer.name "SwinIR-RGB++" --model.use_gradients false --model.grad_weight 4.0

# python train.py fit  -c configs/swin-grad-jpeg.yaml --trainer.devices [0] --trainer.name "SwinIR-Mix" --model.use_gradients true --model.mixed true  --model.grad_weight 4.0 #--model.mode "mix" 

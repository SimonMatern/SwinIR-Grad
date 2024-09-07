#!/bin/bash

# Train on Color Changes (Illumination, brightness, etc.)
#python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [0] --trainer.name "SwinIR3D-RGB++" --model.use_gradients false --model.mixed false --model.grad_weight 4.0 --data.augm "color"
python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [0] --trainer.name "SwinIR3D-Mix" --model.use_gradients true --model.mixed true  --model.grad_weight 4.0 --data.augm "color" &
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [0] --trainer.name "SwinIR3D-RGB" --model.use_gradients false --model.mixed false --model.grad_weight 0.0 --data.augm "color"
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [0] --trainer.name "SwinIR3D-Grad" --model.use_gradients true --model.mixed false --model.mode "grad" --model.grad_weight 4.0 --data.augm "color" &

# Train on blurry images
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-RGB" --model.use_gradients false --model.mixed false --model.grad_weight 0.0 --data.augm "blur"
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-RGB++" --model.use_gradients false --model.mixed false --model.grad_weight 4.0 --data.augm "blur"
python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-Mix" --model.use_gradients true --model.mixed true  --model.grad_weight 4.0 --data.augm "blur" &&
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-Grad" --model.use_gradients true --model.mixed false --model.mode "grad" --model.grad_weight 4.0 --data.augm "blur" &&
fg

# Train on Jpeg Compression
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-RGB" --model.use_gradients false --model.mixed false --model.grad_weight 0.0 --data.augm "jpeg"
# python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-RGB++" --model.use_gradients false --model.mixed false --model.grad_weight 4.0 --data.augm "jpeg"
python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [1] --trainer.name "SwinIR3D-Mix" --model.use_gradients true --model.mixed true  --model.grad_weight 4.0 --data.augm "jpeg" 
#python train.py fit  -c configs/swinIR3D.yaml --trainer.devices [0] --trainer.name "SwinIR3D-Grad" --model.use_gradients true --model.mixed false --model.mode "grad" --model.grad_weight 4.0 --data.augm "jpeg"

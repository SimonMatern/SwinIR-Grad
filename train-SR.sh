#!/bin/bash

python train.py fit  -c configs/config.yaml --trainer.devices [0] --trainer.name "SwinIR-Mix" --model.use_gradients true  --model.grad_weight 4.0 --model.mode "mix" --model.mix true

python train.py fit  -c configs/config.yaml --trainer.devices [0] --trainer.name "SwinIR-RGB" --model.use_gradients false --model.grad_weight 0.0


#python train.py fit  -c config.yaml --trainer.devices [1] --trainer.name "SwinIR-Grad" --model.use_gradients true  --model.grad_weight 4.0 --model.mode "grad" --model.mix false


#python train.py fit  -c config.yaml --trainer.devices [1] --trainer.name "SwinIR-Grad-AVG" --model.use_gradients true  --model.grad_weight 4.0 --model.mode "mean" --model.mix true

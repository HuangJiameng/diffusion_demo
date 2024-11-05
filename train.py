from data import ImageProcessor
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import build_model
from diffusion import create_diffusion
import torch
from time import time
from trainer import Trainer
import yaml

# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a simple diffusion",
        epilog="Example command: python train.py --image_path 'path/to/image.jpg' --output_dir './outputs' --model_type 'DiT' --config_path 'config/config.yml' --epochs 3000 --log_interval 50 --save_interval 300"
    )
    parser.add_argument("--image_path", type=str, help="Path to the input image", default="example.jpg")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs", default="./ag_demo_mlp")
    parser.add_argument("--model_type", type=str, help="Model Type", default="DiT", choices=["DiT", "SimpleMLPAdaLN"])
    parser.add_argument("--config_path", type=str, help="Path of model config (YAML file)", default="default.yml")
    parser.add_argument("--device", type=str, help="device", default="cuda:0")
    parser.add_argument("--epochs", type=int, help="Total number of training epochs", default=3000)
    parser.add_argument("--log_interval", type=int, help="Interval for logging", default=50)
    parser.add_argument("--save_interval", type=int, help="Interval for saving model checkpoints", default=300)
    return parser.parse_args()

# 主函数
def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 使用 ImageProcessor 获取数据集
    image_processor = ImageProcessor()
    X, Y = image_processor.get_dataset(args.image_path)

    device = torch.device(args.device) 

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    model = build_model(args.model_type, config)
    model.to(device)

    diffusion = create_diffusion(timestep_respacing="")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    trainer = Trainer(model, 
                      diffusion = diffusion, 
                      optimizer = optimizer, 
                      device, 
                      X, 
                      Y, 
                      output_dir = args.output_dir, 
                      epochs = args.epochs, 
                      log_interval = args.log_interval, 
                      save_interval = args.save_interval,
                      )
    trainer.train()

    

if __name__ == "__main__":
    args = parse_args()
    main(args)


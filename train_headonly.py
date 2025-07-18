from ultralytics import YOLO
import os
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--data', type=str, default='./data/NW2DI.yaml', help='Path to dataset config file')
    parser.add_argument('--model', type=str, default='./runs/detect/train/weights/best.pt', help='Path to model config file')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    return parser.parse_args()

def print_args(args):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

if __name__ == '__main__':
    args = parse_args()
    print_args(args)

    model_url = Path(args.model) if os.path.exists(args.model) else args.model
    model = YOLO(model_url)

    # train detect head only

    for param in model.model.parameters():
        param.requires_grad = False  # 冻结所有层

    head_layers = model.model.model[-1]  # 假设最后一层是检测头
    for param in head_layers.parameters():
        param.requires_grad = True

    model.train(
        data=Path(args.data),
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        workers=args.workers,
    )


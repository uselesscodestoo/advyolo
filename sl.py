import os
import shutil
import random
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import select_device
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SelfSupervisedStyleTransferTrainer:
    """自监督风格迁移训练器，用于在无标签目标域上微调YOLO模型"""
    
    def __init__(self, 
                 pretrained_model_path='yolov8s.pt',
                 source_data_path=None,  # 源域数据路径（可选）
                 target_data_path=None,  # 目标域数据路径
                 output_dir='runs/self_supervised',
                 device='cuda',
                 num_cycles=5,  # 自训练循环次数
                 conf_threshold=0.5,  # 伪标签置信度阈值
                 iou_threshold=0.6,   # 伪标签NMS的IoU阈值
                 train_ratio=0.8,     # 训练集比例
                 seed=42):
        """初始化训练器"""
        self.pretrained_model_path = pretrained_model_path
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.output_dir = Path(output_dir)
        self.device = select_device(device)
        self.num_cycles = num_cycles
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 创建输出目录
        self.run_dir = self.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录配置
        self._log_config()
        
        # 初始化模型
        self.model = YOLO(Path(pretrained_model_path))
        
        # 获取类别数量（假设源域和目标域类别相同）
        self.num_classes = len(self.model.names)
        
        # 准备目标域数据
        self._prepare_target_data()
    
    def _log_config(self):
        """记录训练配置"""
        config = {
            'pretrained_model_path': self.pretrained_model_path,
            'source_data_path': self.source_data_path,
            'target_data_path': self.target_data_path,
            'num_cycles': self.num_cycles,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }
        
        with open(self.run_dir / 'config.txt', 'w') as f:
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
        
        LOGGER.info(colorstr('配置:') + f" 自监督训练配置已保存到 {self.run_dir / 'config.txt'}")
    
    def _prepare_target_data(self):
        """准备目标域数据"""
        LOGGER.info(colorstr('数据:') + f" 准备目标域数据 from {self.target_data_path}")
        
        # 假设目标数据路径下有images文件夹
        images_path = Path(self.target_data_path) / 'images'
        if not images_path.exists():
            raise FileNotFoundError(f"未找到图像文件夹: {images_path}")
        
        # 创建临时数据目录
        self.temp_data_dir = self.run_dir / '../temp_data'
        if self.temp_data_dir.exists():
            print(f"创建临时数据目录已经完成！")
            self.data_yaml = self.temp_data_dir / 'data.yaml'
            return
        
        print(f"临时数据目录：{self.temp_data_dir}")
        train_dir = self.temp_data_dir / 'train'
        train_dir.mkdir(exist_ok=True)
        val_dir = self.temp_data_dir / 'val'
        val_dir.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(images_path.glob('*.*'))
        if not image_files:
            raise FileNotFoundError(f"在 {images_path} 中未找到图像文件")
        
        LOGGER.info(colorstr('数据:') + f" 找到 {len(image_files)} 张目标域图像")
        
        # 创建数据配置文件
        data_yaml = self.temp_data_dir / 'data.yaml'
        with open(data_yaml, 'w') as f:
            f.write(f"path: {self.temp_data_dir}\n")
            f.write("train: train\n")
            f.write("val: val\n")
            f.write(f"nc: {self.num_classes}\n")
            f.write(f"names: {self.model.names}\n")
        
        # 划分训练集和验证集
        train_files, val_files = train_test_split(
            image_files, 
            train_size=self.train_ratio, 
            random_state=self.seed
        )
        
        LOGGER.info(colorstr('数据:') + f" 训练集: {len(train_files)} 张图像, 验证集: {len(val_files)} 张图像")
        
        # 创建图像目录
        train_images_dir = train_dir / 'images' 
        val_images_dir = val_dir / 'images' 
        train_labels_dir = train_dir / 'labels'
        val_labels_dir = val_dir / 'labels'
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 复制图像到相应目录
        for img_path in train_files:
            shutil.copy(img_path, train_images_dir / img_path.name)
        
        for img_path in val_files:
            shutil.copy(img_path, val_images_dir / img_path.name)
        
        self.data_yaml = data_yaml
        LOGGER.info(colorstr('数据:') + f" 目标域数据准备完成，配置文件: {data_yaml}")
    
    def _generate_pseudo_labels(self, image_dir, output_dir, cycle):
        """生成伪标签"""
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + f" 生成伪标签 from {image_dir} to {output_dir}")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(image_dir.glob('*.*'))
        total_images = len(image_files)
        
        if total_images == 0:
            LOGGER.warning(f"在 {image_dir} 中未找到图像文件")
            return 0
        
        LOGGER.info(f"将为 {total_images} 张图像生成伪标签")
        
        # 初始化计数器
        pseudo_label_count = 0
        
        # 批量处理图像（每次处理一部分，避免内存溢出）
        batch_size = 32  # 可根据内存情况调整
        for i in tqdm(range(0, total_images, batch_size), desc="生成伪标签"):
            batch_files = image_files[i:i+batch_size]
            
            # 使用当前模型进行预测
            results = self.model.predict(
                source=[str(f) for f in batch_files],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                device=self.device
            )
            
            # 处理预测结果并生成伪标签
            for result, img_path in zip(results, batch_files):
                img_path = Path(img_path)
                label_path = output_dir / f"{img_path.stem}.txt"
                
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    # 如果没有检测到物体，创建空标签文件
                    with open(label_path, 'w') as f:
                        pass
                    continue
                
                # 写入YOLO格式的标签
                with open(label_path, 'w') as f:
                    for box in boxes:
                        cls = int(box.cls)
                        # conf = float(box.conf)
                        xywh = box.xywhn[0].tolist()  # 归一化的xywh格式
                        # f.write(f"{cls} {' '.join([str(x) for x in xywh])} {conf}\n")
                        f.write(f"{cls} {' '.join([str(x) for x in xywh])}\n")
                        pseudo_label_count += 1
            
            # 释放内存
            del results
            torch.cuda.empty_cache()
        
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + f" 生成了 {pseudo_label_count} 个伪标签")
        return pseudo_label_count

    def _generate_val_labels(self, image_dir, lable_dir, output_dir):
        """生成验证集标签"""
        for img_path in image_dir.iterdir():
            if img_path.suffix in ['.jpg', '.png']:
                lable_file = f"{img_path.stem}.txt"
                shutil.copy(lable_dir / lable_file, output_dir / lable_file)

    
    def _train_with_pseudo_labels(self, cycle):
        """使用伪标签训练模型"""
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + " 开始使用伪标签训练模型")
        
        # 训练模型
        results = self.model.train(
            data=str(self.data_yaml),
            epochs=10,  # 每轮训练的轮次
            batch=8,
            imgsz=512,
            lr0=0.001,
            device=self.device,
            workers=1,
        )
        
        # 保存当前循环的模型
        cycle_model_path = self.run_dir / f'cycle_{cycle}' / 'weights' / 'best.pt'
        self.current_model_path = cycle_model_path
        
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + f" 模型训练完成，保存到 {cycle_model_path}")
        return results
    
    def _evaluate_model(self, cycle):
        """评估模型性能"""
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + " 评估模型性能")
        
        # 加载当前模型
        model = YOLO(self.current_model_path)
        
        # 在验证集上评估
        metrics = model.val(
            data=str(self.data_yaml),
            batch=8,
            imgsz=512,
            device=self.device
        )
        
        # 记录评估结果
        results_path = self.run_dir / f'cycle_{cycle}' / 'evaluation_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"Cycle {cycle} evaluation results:\n")
            f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            for i, cls_name in enumerate(self.model.names):
                f.write(f"Class {cls_name} AP@0.5: {metrics.box.ap50[i]:.4f}\n")
        
        LOGGER.info(colorstr(f'循环 {cycle}/{self.num_cycles}:') + f" 评估完成，结果保存到 {results_path}")
        return metrics
    
    def run(self):
        """运行自监督风格迁移训练"""
        LOGGER.info(colorstr('开始:') + f" 自监督风格迁移训练，共 {self.num_cycles} 个循环")

        train_image_dir = self.temp_data_dir / 'train' / 'images'
        train_labels_dir = self.temp_data_dir / 'train' / 'labels'
        val_image_dir = self.temp_data_dir / 'val' / 'images' 
        val_labels_dir = self.temp_data_dir / 'val' / 'labels'

        self._generate_val_labels(val_image_dir, Path(self.target_data_path) / 'labels', val_labels_dir)
        for cycle in range(1, self.num_cycles + 1):
            LOGGER.info("=" * 80)
            LOGGER.info(colorstr('循环') + f" [{cycle}/{self.num_cycles}] 开始")
            
            # 1. 生成伪标签
            self._generate_pseudo_labels(train_image_dir, train_labels_dir, cycle)
            
            # 2. 使用伪标签训练模型
            self._train_with_pseudo_labels(cycle)
            
            # 3. 评估模型
            # metrics = self._evaluate_model(cycle)
            
            LOGGER.info(colorstr('循环') + f" [{cycle}/{self.num_cycles}] 完成")
            LOGGER.info("=" * 80)
        
        LOGGER.info(colorstr('完成:') + f" 自监督风格迁移训练完成，最终模型保存在 {self.current_model_path}")
        return self.current_model_path

# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'pretrained_model_path': 'runs/detect/train7/weights/best.pt',  # 预训练模型路径
        'target_data_path': 'H:/work/ml/object_detect/DIOR',  # 目标域数据路径
        'output_dir': 'runs/self_supervised',  # 输出目录
        'num_cycles': 5,  # 自训练循环次数
        'conf_threshold': 0.5,  # 伪标签置信度阈值
        'iou_threshold': 0.6,  # 伪标签NMS的IoU阈值
        'train_ratio': 0.8,  # 训练集比例
        'seed': 42  # 随机种子
    }
    
    # 创建并运行训练器
    trainer = SelfSupervisedStyleTransferTrainer(**config)
    final_model_path = trainer.run()
    
    print(f"自监督训练完成！最终模型保存在: {final_model_path}")
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml, print_args
import os
from pathlib import Path
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
RANK = int(os.getenv("RANK", -1))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "yolov11/data/NW2DI.yaml", help="dataset.yaml path")
    parser.add_argument("--model", type=str, default=ROOT / "runs/detect/train7/weights/best.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=2, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt

# Argument	Type	Default	Description
# data	str	None	Specifies the path to the dataset configuration file (e.g., coco8.yaml). This file includes paths to validation data, class names, and number of classes.
# imgsz	int	640	Defines the size of input images. All images are resized to this dimension before processing. Larger sizes may improve accuracy for small objects but increase computation time.
# batch	int	16	Sets the number of images per batch. Higher values utilize GPU memory more efficiently but require more VRAM. Adjust based on available hardware resources.
# save_json	bool	False	If True, saves the results to a JSON file for further analysis, integration with other tools, or submission to evaluation servers like COCO.
# conf	float	0.001	Sets the minimum confidence threshold for detections. Lower values increase recall but may introduce more false positives. Used during validation to compute precision-recall curves.
# iou	float	0.7	Sets the Intersection Over Union threshold for Non-Maximum Suppression. Controls duplicate detection elimination.
# max_det	int	300	Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections and manage computational resources.
# half	bool	True	Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy.
# device	str	None	Specifies the device for validation (cpu, cuda:0, etc.). When None, automatically selects the best available device. Multiple CUDA devices can be specified with comma separation.
# dnn	bool	False	If True, uses the OpenCV DNN module for ONNX model inference, offering an alternative to PyTorch inference methods.
# plots	bool	False	When set to True, generates and saves plots of predictions versus ground truth, confusion matrices, and PR curves for visual evaluation of model performance.
# classes	list[int]	None	Specifies a list of class IDs to train on. Useful for filtering out and focusing only on certain classes during evaluation.
# rect	bool	True	If True, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency by processing images in their original aspect ratio.
# split	str	'val'	Determines the dataset split to use for validation (val, test, or train). Allows flexibility in choosing the data segment for performance evaluation.
# project	str	None	Name of the project directory where validation outputs are saved. Helps organize results from different experiments or models.
# name	str	None	Name of the validation run. Used for creating a subdirectory within the project folder, where validation logs and outputs are stored.
# verbose	bool	False	If True, displays detailed information during the validation process, including per-class metrics, batch progress, and additional debugging information.
# save_txt	bool	False	If True, saves detection results in text files, with one file per image, useful for further analysis, custom post-processing, or integration with other systems.
# save_conf	bool	False	If True, includes confidence values in the saved text files when save_txt is enabled, providing more detailed output for analysis and filtering.
# workers	int	8	Number of worker threads for data loading. Higher values can speed up data preprocessing but may increase CPU usage. Setting to 0 uses main thread, which can be more stable in some environments.
# augment	bool	False	Enables test-time augmentation (TTA) during validation, potentially improving detection accuracy at the cost of inference speed by running inference on transformed versions of the input.
# agnostic_nms	bool	False	Enables class-agnostic Non-Maximum Suppression, which merges overlapping boxes regardless of their predicted class. Useful for instance-focused applications.
# single_cls	bool	False	Treats all classes as a single class during validation. Useful for evaluating model performance on binary detection tasks or when class distinctions aren't important.

def val_args_filter(opt: dict):
    Identifiable = ["data", "imgsz", "batch", "save_json",
                    "conf", "iou", "max_det", "half", 
                    "device", "dnn", "plots", "classes", 
                    "rect", "split", "project", "name", 
                    "verbose", "save_txt", "save_conf", 
                    "workers", "augment", "agnostic_nms", 
                    "single_cls"]
    filted = {}
    for k, v in opt.items():
        for i in Identifiable:
            if i in k:
                filted[i] = v
    return filted

if __name__ == "__main__":
    opt = parse_args()
    args = val_args_filter(vars(opt))
    model = YOLO(Path(opt.model))
    print_args(args)
    results = model.val(**args)
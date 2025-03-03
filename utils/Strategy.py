import yaml
import torch
from ultralytics import YOLO
from abc import ABC, abstractmethod

# 1. 读取 YAML 配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# 2. 定义策略接口
class YOLOStrategy(ABC):
    @abstractmethod
    def get_model(self):
        pass

# 3. 具体策略 - YOLOv5
class YOLOv5Strategy(YOLOStrategy):
    def get_model(self):
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 4. 具体策略 - YOLOv8
class YOLOv8Strategy(YOLOStrategy):
    def get_model(self):
        return YOLO("yolov8n.pt")  # YOLOv8 官方模型

# 5. 具体策略 - YOLOv11
class YOLOv11Strategy(YOLOStrategy):
    def get_model(self):
        return YOLO("yolo11n.pt")  # YOLOv11 模型

# 6. 具体策略 - YOLOv12
class YOLOv12Strategy(YOLOStrategy):
    def get_model(self):
        return YOLO("yolo12n.pt")  # YOLOv12 模型

# 7. 选择策略
class YOLOSelector:
    _strategies = {
        "yolov5": YOLOv5Strategy(),
        "yolov8": YOLOv8Strategy(),
        "yolov11": YOLOv11Strategy(),
        "yolov12": YOLOv12Strategy(),
    }

    @staticmethod
    def get_model(version):
        strategy = YOLOSelector._strategies.get(version)
        if strategy:
            return strategy.get_model()
        else:
            raise ValueError(f"不支持的 YOLO 版本: {version}")

# 8. 运行
if __name__ == "__main__":
    config = load_config()
    yolo_version = config.get("yolo_version", "yolov8")  # 默认使用 YOLOv8

    try:
        model = YOLOSelector.get_model(yolo_version)
        print(f"当前选择的模型: {model}")
    except ValueError as e:
        print(e)

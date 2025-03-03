
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import os

from pathlib import Path
import cv2
import torch

import paddlehub as hub

import utils.Strategy as Strategy
import utils.vehicle as Vehicle
import yaml


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def yolov_inference(image, video, model_id, image_size, conf_threshold):
    yolo = Strategy.YOLOSelector.get_model(model_id)
    if image is not None:
        return yolo(image)
    else:
        return yolo(video)

def bbox_rel(*xyxy):
    """" 计算边界值 """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    增加不同颜色边框
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # 编号
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

# def detect_license_plate(vehicle_region):
#     """
#     车牌检测（可以用训练好的 YOLO 车牌检测模型，或者 OpenCV 方式）
#     :param vehicle_region: 车辆裁剪区域
#     :return: 车牌区域 (x1, y1, x2, y2)
#     """
#     # TODO: 这里可以换成 YOLO 车牌检测模型
#     gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / float(h)
#         if 2 < aspect_ratio < 5:  # 车牌通常是矩形
#             return (x, y, x + w, y + h)
#
#     return None  # 没找到车牌

def detector(source, model_id, image_size=640, conf_threshold=0.4,
             deepsort_cfg="deep_sort_pytorch/configs/deep_sort.yaml"):
    """
    使用 YOLOv10 进行目标检测，并使用 DeepSORT 进行目标跟踪。

    :param source: 图像或视频路径
    :param model_id: YOLOv10 模型权重路径
    :param image_size: 输入图像尺寸
    :param conf_threshold: 置信度阈值
    :param deepsort_cfg: DeepSORT 配置文件路径
    :return: 处理后的视频/图像
    """
    # 初始化 DeepSORT
    cfg = get_config()
    cfg.merge_from_file(deepsort_cfg)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=torch.cuda.is_available())

    # 读取输入源（图片或视频）
    is_video = source.endswith(('.mp4', '.avi', '.mov'))
    cap = cv2.VideoCapture(source) if is_video else None
    output_dir = yaml.safe_load(open("config.yaml"))['output_dir']

    if is_video:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(output_dir, f"output_{Path(source).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Error: VideoWriter failed to open!")
            return

    frame_count = 0
    count =  5
    ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        else:
            frame = cv2.imread(source)

        # YOLOv 目标检测
        detections = yolov_inference(None if not is_video else frame, None if is_video else frame, model_id, image_size, conf_threshold)

        bbox_xywh = []
        confs = []

        if detections and detections[0].boxes is not None:
            # 获取边界框信息
            boxes = detections[0].boxes.xywh.cpu().numpy()  # (x, y, w, h)
            conf_scores = detections[0].boxes.conf.cpu().numpy()  # 置信度
            classes = detections[0].boxes.cls.cpu().numpy()  # 类别索引

            for i in range(len(boxes)):
                bbox_xywh.append(boxes[i].tolist())  # 转为 Python 列表
                confs.append([float(conf_scores[i])])  # 置信度转换为浮点数并封装成列表

        # 转换为 Tensor
        if bbox_xywh:
            xywhs = torch.tensor(bbox_xywh, dtype=torch.float32).view(-1, 4)
            confss = torch.tensor(confs, dtype=torch.float32).view(-1, 1)
        else:
            xywhs = torch.empty((0, 4), dtype=torch.float32)
            confss = torch.empty((0, 1), dtype=torch.float32)

        # DeepSORT 进行目标跟踪
        outputs = deepsort.update(xywhs, confss, frame)

        # 绘制跟踪结果
        for output in outputs:
            x1, y1, x2, y2, track_id = output[:5]

            # 画出车辆跟踪框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 如果 track_id 不在 VEHICLES 字典中，则创建新对象
            if track_id not in Vehicle.VEHICLES:
                Vehicle.VEHICLES[track_id] = Vehicle.Vehicle(track_id, None, bbox_rel(x1, y1, x2, y2))
                print(f"New vehicle detected: ID {track_id}")

            # ---- 车牌识别部分 ----
            vehicle_region = frame[y1:y2, x1:x2]  # 裁剪车辆区域vehicle_region = frame[y1:y2, x1:x2]  # 裁剪车辆区域
            plate_coords = x1, y1, x2, y2  # 车牌检测

            if plate_coords:
                px1, py1, px2, py2 = plate_coords
                plate_region = vehicle_region[py1:py2, px1:px2]  # 裁剪车牌区域

                if plate_region.size > 0:
                    ocr_result = ocr.recognize_text(images=[plate_region], visualization=False)
                    plate_number = "".join([d['text'] for res in ocr_result for d in res['data']])

                    if plate_number:
                        Vehicle.VEHICLES[track_id].plate_number = plate_number  # 赋值车牌号
                        print(f"Vehicle {track_id} - Plate: {plate_number}")


        cv2.imshow("Tracking", frame)

        #保存预测结果
        count -= 1
        if is_video:
            print(f"Processing frame {frame_count}/{total_frames}")
            out.write(frame)
        elif count == 0:
            cv2.imwrite(f"{output_dir}/output_{Path(source).stem}.jpg", frame)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_video:
        cap.release()
        out.release()
        print(f"Processed video saved at {output_path}")
    cv2.destroyAllWindows()

def getTextFromOCR(results):
    textList = []
    for result in results:
        for text in result['data']:
            textList.append(text['text'])
    return textList

if __name__ == "__main__":
    ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)
    ocr_result = ocr.recognize_text(images=[cv2.imread('res/input/图片1.png')], visualization=False, )
    ocr_text = getTextFromOCR(ocr_result)
    print(ocr_text)
    # detector('res/input/图片1.png', 'yolov12')





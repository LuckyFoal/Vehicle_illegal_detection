
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import os
import time
import threading


from pathlib import Path
import cv2
import torch

import paddlehub as hub

import utils.Strategy as Strategy
import utils.vehicle as Vehicle
import utils.algorithm as Algorithm
import utils.test as Test
from utils.vehicle import ILLEGAL_TRACKS as illegal_track
from utils.vehicle import VEHICLES as VEHICLES
import yaml

# 读取配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEEPSORT_CFG = config.get("deepsort_cfg", "deep_sort_pytorch/configs/deep_sort.yaml")
OUTPUT_DIR = config.get("output_dir", "./output")

# 初始化 OCR 车牌识别模型
ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)

# 违停区域 (x1, y1, x2, y2)
NO_PARKING_ZONE = (100, 200, 1500, 2000)  # 这里需要根据实际情况修改

PERFORM_CHECK = False


def yolov_inference(image, video, model_id, image_size = 640, conf_threshold = 0.4):
    yolo = Strategy.YOLOSelector.get_model(model_id)
    # print(yolo.names) # 获取模型类别索引
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

def init_deepsort():
    """初始化 DeepSORT 目标跟踪器"""
    cfg = get_config()
    cfg.merge_from_file(DEEPSORT_CFG)
    return DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available())

def load_video_source(source):
    """加载视频或图像源"""
    is_video = source.endswith(('.mp4', '.avi', '.mov'))
    cap = cv2.VideoCapture(source) if is_video else None
    return cap, is_video

def detect_objects(frame, model_id, image_size, conf_threshold):
    """使用 YOLO 进行目标检测"""
    detections = yolov_inference(frame, frame, model_id, image_size, conf_threshold)
    bbox_xywh, confs = [], []
    CAR_CLASS_IDS = {1, 2, 3, 5, 7}  # 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',....

    if detections and detections[0].boxes is not None:
        boxes = detections[0].boxes.xywh.cpu().numpy()  # (x, y, w, h)
        conf_scores = detections[0].boxes.conf.cpu().numpy()
        classes = detections[0].boxes.cls.cpu().numpy().astype(int)  # 类别索引

        for i in range(len(boxes)):
            if classes[i] in CAR_CLASS_IDS:  # 只保留车辆类别
                bbox_xywh.append(boxes[i].tolist())  # 转为 Python 列表
                confs.append([float(conf_scores[i])])  # 置信度转换为浮点数并封装成列表

    if bbox_xywh:
        xywhs = torch.tensor(bbox_xywh, dtype=torch.float32).view(-1, 4)
        confss = torch.tensor(confs, dtype=torch.float32).view(-1, 1)
    else:
        xywhs = torch.empty((0, 4), dtype=torch.float32)
        confss = torch.empty((0, 1), dtype=torch.float32)

    return xywhs, confss

def track_objects(deepsort, xywhs, confss, frame):
    """使用 DeepSORT 进行目标跟踪"""
    return deepsort.update(xywhs, confss, frame)


def detect_plate(vehicle_region):
    """
    车牌检测：输入车辆区域，返回车牌坐标（px1, py1, px2, py2）。

    :param vehicle_region: 车辆裁剪区域
    :return: (px1, py1, px2, py2) 车牌坐标
    """
    # # 这里需要实现车牌检测模型，目前假设返回整个车辆区域作为车牌
    h, w, _ = vehicle_region.shape
    # return 0, int(0.6 * h), w, h  # 简单模拟车牌区域
    model = Strategy.YOLOSelector.get_model('yolov11n-plate')
    result =  model.predict(vehicle_region)
    if result and result[0].boxes is not None:
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            return int(x1), int(y1), int(x2), int(y2)
    return 0, int(0.6 * h), w, h  # 简单模拟车牌区域


def recognize_plate_text(plate_region):
    """
    OCR 识别车牌号

    :param plate_region: 车牌区域的图像
    :return: 识别到的车牌号字符串
    """
    if plate_region.size > 0:
        ocr_result = ocr.recognize_text(images=[plate_region], visualization=False)
        return "".join([d['text'] for res in ocr_result for d in res['data']])
    return None

def process_frame(frame, deepsort, model_id, image_size, conf_threshold):
    """处理单帧图像，包括目标检测、跟踪和车牌识别"""
    xywhs, confss = detect_objects(frame, model_id, image_size, conf_threshold)
    outputs = track_objects(deepsort, xywhs, confss, frame)

    for output in outputs:
        x1, y1, x2, y2, track_id = output[:5]

        if track_id not in Vehicle.VEHICLES:
            Vehicle.VEHICLES[track_id] = Vehicle.Vehicle(track_id, None, (x1, y1, x2, y2))

        if PERFORM_CHECK:  # 违停检测仅每秒执行一次
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            vehicle_position = (x1, y1, x2, y2)
            in_no_parking_zone = (
                    NO_PARKING_ZONE[0] <= center_x <= NO_PARKING_ZONE[2] and
                    NO_PARKING_ZONE[1] <= center_y <= NO_PARKING_ZONE[3]
            )

            if in_no_parking_zone:
                if track_id not in illegal_track:
                    illegal_track[track_id] = {"preposition": vehicle_position, "stop_time": 0}
            else:
                illegal_track.pop(track_id, None)  # 车辆离开禁停区域

            # 车牌检测每秒执行一次
        if PERFORM_CHECK:
            vehicle_region = frame[y1:y2, x1:x2]  # 裁剪车辆区域
            plate_coords = detect_plate(vehicle_region)  # 车牌检测
            if plate_coords:
                px1, py1, px2, py2 = plate_coords
                plate_region = vehicle_region[py1:py2, px1:px2]  # 裁剪车牌区域
                plate_number = recognize_plate_text(plate_region)  # 车牌OCR识别
                if plate_number:
                    Vehicle.VEHICLES[track_id].plate_number = plate_number  # 赋值车牌号
                    print(f"Vehicle {track_id} - Plate: {plate_number}")

        # 绘制检测框和车牌号
        cv2.rectangle(frame, NO_PARKING_ZONE[0:2], NO_PARKING_ZONE[2:4], (255, 0, 0), 3) #绘制违停区域
        if VEHICLES[track_id] and VEHICLES[track_id].isIllegal():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f'Illegal', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f'ID: {track_id} Plate: {Vehicle.VEHICLES[track_id].plate_number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def illegal_track_detection():
    violated_vehicles = []

    for track_id, info in list(illegal_track.items()):
        latest_position = Vehicle.VEHICLES[track_id].getLocation()

        if not latest_position:
            illegal_track.pop(track_id, None)
            continue

        if not Algorithm.is_vehicle_moving(latest_position, info["preposition"]):
            info["stop_time"] += 1
        else:
            info["stop_time"] = 0
            info["preposition"] = latest_position

        if info["stop_time"] >= 1:
            violated_vehicles.append(track_id)
            VEHICLES[track_id].setIllegal(True)

    if violated_vehicles:
        print(f"⚠️ 违停车辆: {violated_vehicles}")


def detector(source, model_id, image_size=640, conf_threshold=0.4):
    """主函数：读取视频/图像，进行目标检测、跟踪和车牌识别"""
    deepsort = init_deepsort()
    cap, is_video = load_video_source(source)

    if is_video:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(OUTPUT_DIR, f"output_{Path(source).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Error: VideoWriter failed to open!")
            return

    frame_count = 0
    count = 5
    last_detection_time = 0 # 记录上次调用违停检测的时间（单位：秒）
    global PERFORM_CHECK

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        else:
            frame = cv2.imread(source)

        # 每隔 1 秒进行一次违停检测（基于视频的时间）
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 # 获取当前帧的时间（秒）
        if frame_time - last_detection_time >= 1:
            illegal_track_detection()
            last_detection_time = frame_time
            PERFORM_CHECK = True

        frame = process_frame(frame, deepsort, model_id, image_size, conf_threshold)
        PERFORM_CHECK = False
        cv2.imshow("Tracking", frame)
        count -= 1
        if is_video:
            print(f"Processing frame {frame_count}/{total_frames}")
            out.write(frame)
        elif count == 0:
            cv2.imwrite(f"{OUTPUT_DIR}/output_{Path(source).stem}.jpg", frame)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_video:
        cap.release()
        out.release()
        print(f"Processed video saved at {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test.test()
    start_time = time.time()
    detector('res/input/short.mp4', 'yolov12')
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

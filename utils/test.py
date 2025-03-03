
from pathlib import Path
import cv2

import paddlehub as hub
import utils.Strategy as Strategy
def yolov_inference(image, video, model_id, image_size = 640, conf_threshold = 0.4):
    yolo = Strategy.YOLOSelector.get_model(model_id)
    # print(yolo.names) # 获取模型类别索引
    if image is not None:
        return yolo(image)
    else:
        return yolo(video)

def show_yolov_result(result):
    """
    绘制 YOLOv 检测结果

    :param result: YOLOv 检测返回的结果对象
    :return: 带有检测框的图像
    """
    if not result or not isinstance(result, list):
        print("Error: Invalid detection result format.")
        return None

    detections = result[0]  # 取出第一个检测结果
    frame = detections.orig_img if hasattr(detections, 'orig_img') else None  # 获取原始图像

    if detections and detections.boxes is not None:
        boxes = detections.boxes.xyxy.cpu().numpy()  # 获取 (x1, y1, x2, y2) 坐标
        confs = detections.boxes.conf.cpu().numpy()  # 获取置信度
        classes = detections.boxes.cls.cpu().numpy().astype(int)  # 获取类别索引

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])  # 取整
            label = f"{classes[i]}: {confs[i]:.2f}"  # 显示类别索引和置信度

            # 画矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示类别索引和置信度
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def getTextFromOCR(results):
    textList = []
    for result in results:
        for text in result['data']:
            textList.append(text['text'])
    return textList
def test():
    image_path = "res/input/car.png"
    image = cv2.imread(image_path)
    result = yolov_inference(image_path, None, "yolov11n-plate")

    if result and isinstance(result, list) and hasattr(result[0], 'boxes'):
        detections = result[0]
        boxes = detections.boxes.xyxy.cpu().numpy()  # 获取检测框 (x1, y1, x2, y2)

        plate_regions = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate_region = image[y1:y2, x1:x2]  # 裁剪车牌区域
            cv2.imshow("Plate", plate_region)
            # 保存车牌图片
            cv2.imwrite(f"res/output/plate_{Path(image_path).stem}.jpg", plate_region)
            cv2.waitKey(10000)
            if plate_region.size > 0:
                plate_regions.append(plate_region)
    else:
        plate_regions = []

    ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)
    ocr_result = ocr.recognize_text(images=[cv2.imread('res/input/plate_car.jpg')], visualization=False, )
    ocr_text = getTextFromOCR(ocr_result)
    print(ocr_text)

    #绘制预测结果
    frame = show_yolov_result(result)
    cv2.imshow("YOLOv Detection", frame)
    cv2.waitKey(10000)

    cv2.destroyAllWindows()
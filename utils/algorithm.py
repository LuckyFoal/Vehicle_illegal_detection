import math

def is_vehicle_moving(current_position, previous_position, threshold_ratio=0.05):
    """
    判断车辆是否移动，考虑目标框的大小进行归一化计算

    :param current_position: 当前帧的车辆位置 (x1, y1, x2, y2)
    :param previous_position: 上一帧的车辆位置 (x1, y1, x2, y2)
    :param threshold_ratio: 允许的移动比例，基于目标框对角线计算
    :return: True 表示移动，False 表示静止
    """
    if not previous_position:
        return True  # 第一次检测，默认为移动

    # 计算目标框的中心点
    cx, cy = (current_position[0] + current_position[2]) / 2, (current_position[1] + current_position[3]) / 2
    pre_cx, pre_cy = (previous_position[0] + previous_position[2]) / 2, (previous_position[1] + previous_position[3]) / 2

    # 计算欧几里得距离
    distance = math.sqrt((cx - pre_cx) ** 2 + (cy - pre_cy) ** 2)

    # 计算当前目标框的对角线长度
    w = current_position[2] - current_position[0]
    h = current_position[3] - current_position[1]
    diagonal_length = math.sqrt(w**2 + h**2)

    # 计算移动阈值：目标框对角线的 threshold_ratio 倍
    movement_threshold = diagonal_length * threshold_ratio

    return distance > movement_threshold  # 若距离大于阈值，表示移动

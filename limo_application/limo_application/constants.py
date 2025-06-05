# 주변 환경에 덜 민감하게 (자율주행 도로에만 집중하게) 이미지 위쪽을 자르는 비율
crop_top_ratio = 0.5  # 0.0 ~ 1.0

# 사진 저장 간격 (sec)
save_interval = 0.1

# 키보드 입력값과 대응되는 action 맵
keymap = {
    'u': { # left
        'linear_x': 0.5,
        'angular_z': 1.0
    },
    'i': { # forward
        'linear_x': 0.7,
        'angular_z': 0.0
    },
    'o': { # right
        'linear_x': 0.5,
        'angular_z': -1.0
    }
}

# 데이터 수집 폴더 경로
collect_dir = '/home/jetson/yolo_classification/dataset/collect'  


# predict 예측 간격 시간 (초)
prediction_interval = 0.1  

# predict 할때 앞쪽 물체가 너무 가까이 있는 경우 중단을 위한 거리 (meters)
forward_object_distance_threshold = 0.4

# 확률 분포를 통해해
smooth_action = False


# 주변 환경에 덜 민감하게 (자율주행 도로에만 집중하게) 이미지 위쪽을 자르는 비율
crop_top_ratio = 0.5  # 0.0 ~ 1.0

# 키보드 입력값과 대응되는 action 맵 (소문자 영어로)
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

###### collect #######
collect_dir = '/home/jetson/yolo_classification/dataset/collect'  # 데이터 수집 폴더
save_interval = 0.2 # seconds, 수집 주기


###### predict #######
predict_model_path = '/home/jetson/yolo_classification/runs/classify/train/weights/best.pt'
action_smooth = False    # True: 평균으로 부드럽게 주행, False: Best action으로 주행
prediction_interval = 0.1  # seconds
forward_object_distance_threshold = 1


# limo_application

## build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select limo_application
source install/setup.bash
```

## launch

### limo_start (필수)
- 필수 로봇 시작
```bash
ros2 launch limo_bringup limo_start.launch.py 
```

### collect
- 데이터 수집
```bash
ros2 run limo_application collect 
```

### predict
- 예측 서버 실행
```bash
docker run -it --rm \
    --name yolo_server \
    --ipc=host \
    -p 5000:5000 \
    --runtime=nvidia \
    -v /home/jetson/yolo_classification:/workspace \
    yolo:latest \
    python3 server.py
```

- 예측 및 자율주행
```bash
ros2 run limo_application predict 
```
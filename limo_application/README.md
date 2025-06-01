# limo_application

## build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select limo_application
source install/setup.bash
```

## launch

- limo_start
    ```bash
    ros2 launch limo_bringup limo_start.launch.py 
    ```

- collect
    ```bash
    ros2 run limo_application collect 
    ```

- predict
    ```bash
    ros2 run limo_application predict 
    ```
#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import threading
from datetime import datetime
import sys
import termios
import tty
import select
import time
import numpy as np

from limo_application.constants import (
    keymap, 
    collect_dir,
    save_interval,
    crop_top_ratio,
    predict_model_path,
    action_smooth,
    prediction_interval,
    forward_object_distance_threshold
)

###### meta key #######
KEY_UP    = '\x1b[A'   # ↑
KEY_DOWN  = '\x1b[B'   # ↓
KEY_RIGHT = '\x1b[C'   # →
KEY_LEFT  = '\x1b[D'   # ←
KEY_SPACE = ' '        # 스페이스바(정지)
KEY_ENTER = '\r'

class Classifier(Node):
    def __init__(self, mode='collect'):
        super().__init__('classifier_node')    
        self.mode = mode
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Individual subscribers with separate callbacks
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            10
        )

        # cmd_vel publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # latest data
        self.latest_scan = None
        self.latest_image = None
        self.last_save_time = 0
        self.last_key = KEY_SPACE
        self.save_interval = save_interval  # Save interval in seconds
        
        # Control variables
        self.running = True
        
        # Terminal settings for raw input
        self.old_settings = None
        self.setup_terminal()
        
        
        # Check if keymap is empty
        if 'h' in keymap:
            self.get_logger().info("'h' key is reserved for help, It will be ignored in keymap.")
            del keymap['h']
        if 's' in keymap:
            self.get_logger().info("'s' key is reserved for status, It will be ignored in keymap.")
            del keymap['s']
        if 'q' in keymap:
            self.get_logger().info("'q' key is reserved for quit, It will be ignored in keymap.")
            del keymap['q']

        # Keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()



        # mode setup
        self.output_dir = collect_dir
        if self.mode == 'collect':
            self.setup_output_directories()
            self.get_logger().info('Data Collector initialized.')
            self.get_logger().info('Press a number key to control robot (NO ENTER needed).')
            self.get_logger().info('Press "q" to quit, "h" for help, "s" for status.')
        elif self.mode == 'predict':
            if not os.path.exists(predict_model_path):
                raise FileNotFoundError(f'Predict model path does not exist: {predict_model_path}')
            # load model
            from ultralytics import YOLO
            self.model = YOLO(predict_model_path)
            # warmup model
            self.get_logger().info('Warming up model...')
            [self.model(np.zeros((128, 128, 3), dtype=np.uint8)) for _ in range(3)] 
            self.get_logger().info('Predict initialized.')
            self.get_logger().info('Press a number key to stop prediction robot.')
            self.get_logger().info('Press "q" to quit, "h" for help, "s" for status.')
        else:
            raise ValueError(f'Invalid mode: {mode}. Use "collect" or "predict".')

        # Timer for periodic prediction
        if self.mode == 'predict':
            self.predict_timer = self.create_timer(prediction_interval, self.predict_action)


    def predict_action(self):
        if self.get_forward_object_distance():
            self.get_logger().info('Forward object detected!!')
            self.publish_cmd_vel(0,0)
            return
        elif self.last_key == KEY_SPACE:
            self.get_logger().info('KEY_SPACE pressed!!')
            self.publish_cmd_vel(0,0)
            return

        self.predict(publish_cmd_vel=True)

    def get_forward_object_distance(self):
        if self.latest_scan :
            latest_scan_len = len(self.latest_scan)
            last_scan_section_len = latest_scan_len // 3
            right_list = self.latest_scan[:last_scan_section_len]
            # right_mean = np.mean(right_list) if right_list else 0.0
            forward_list = self.latest_scan[last_scan_section_len:2*last_scan_section_len]
            # forward_mean = np.mean(forward_list) if forward_list else 0.0
            smallest = sorted(x for x in forward_list if x != 0)[0]
            left_list = self.latest_scan[2*last_scan_section_len:]
            # left_mean = np.mean(left_list) if left_list else 0.0
            if smallest < forward_object_distance_threshold:
                self.get_logger().info(f'Forward object detected: {smallest:.2f} m')
                return True
        return False

    def predict(self, publish_cmd_vel=False):
        """Run prediction on the latest image"""
        if self.latest_image is None:
            self.get_logger().warn('No image available for prediction')
            return
        
        try:
            results = self.model(
                source=self.img_preprocess(self.latest_image),  # Preprocess image
                # verbose=False,  # 자세한 출력 제거
                # augment=False,  # 분류 모델에서 불필요한 출력 제거
                # imgsz=160
            )
            result = results[0]
            ### 공통 데이터 정보 (result)
            result_orig_img = result.orig_img # 원본 이미지 행렬 배열
            result_orig_shape = result.orig_shape # 원본 이미지
            result_names = result.names  # 클래스 인덱스(클래스 이름 딕셔너리)
            ### 확률 데이터 정보 (result.probs)
            result_probs = result.probs.data.cpu().numpy()

            ### 가장 높은 확률의 클래스 정보
            best_index = result_probs.argmax()
            best_name = result_names[best_index]
            best_prob = result_probs[best_index]
            print(f"best_name: {best_name}, best_prob: {best_prob:.3f}")
            self.get_logger().info(f'Predicted: {best_name} with probability {best_prob:.3f}')
            
            ### sooth action
            linear_x_sum = 0
            angular_z_sum = 0
            for class_idx, key in result_names.items():
                linear_x_sum += result_probs[class_idx] * keymap[key]['linear_x']
                angular_z_sum += result_probs[class_idx] * keymap[key]['angular_z']
            
            linear_x_mean = linear_x_sum / len(result_names)
            angular_z_mean = angular_z_sum / len(result_names)
            self.get_logger().info(f'Smoothed Action - Linear: {linear_x_mean:.2f}, Angular: {angular_z_mean:.2f}')
            
            ### best action
            linear_x_best = keymap[best_name]['linear_x']
            angular_z_best = keymap[best_name]['angular_z']
            self.get_logger().info(f'Best Action - Linear: {linear_x_best:.2f}, Angular: {angular_z_best:.2f}')
            
            ### publish
            if publish_cmd_vel:
                if action_smooth:
                    self.publish_cmd_vel(
                        linear_x=linear_x_mean,
                        angular_z=angular_z_mean,
                        save_data=False
                    )
                else:
                    self.publish_cmd_vel(
                        linear_x=linear_x_best,
                        angular_z=angular_z_best,
                        save_data=False
                    )
                self.get_logger().info('Published CMD_VEL based on prediction')
        except Exception as e:
            self.get_logger().error(f'Error during prediction: {str(e)}')

    def setup_terminal(self):
        """Setup terminal for raw input (no enter required)"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            self.get_logger().error(f'Error setting up terminal: {str(e)}')
            self.old_settings = None

    def restore_terminal(self):
        """Restore terminal settings"""
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception as e:
                self.get_logger().error(f'Error restoring terminal: {str(e)}')

    def setup_output_directories(self):
        """Create output directories for each keymap entry"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.get_logger().info(f'Created main output directory: {self.output_dir}')
            
            for key in keymap.keys():
                key_dir = os.path.join(self.output_dir, key)
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)
                    self.get_logger().info(f'Created directory: {key_dir}')
        except Exception as e:
            self.get_logger().error(f'Error creating directories: {str(e)}')

    def cmd_vel_callback(self, msg: Twist):
        """Callback for cmd_vel messages"""
        pass

    def scan_callback(self, msg: LaserScan):
        """Callback for laser scan data"""
        self.latest_scan = msg.ranges
    
    def image_callback(self, msg: Image):
        """Individual callback for image topic"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def keyboard_listener(self):
        """Listen for keyboard input in a separate thread - instant response"""
        self.get_logger().info('Keyboard listener started. Press keys directly (no Enter needed)...')
        
        while self.running and rclpy.ok():
            try:
                # Check if input is available using select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    self.last_key = char
                    if char:
                        if char == '\x03':  # Ctrl+C
                            self.get_logger().info('Ctrl+C received')
                            self.running = False
                            break
                        elif (char in keymap) or (char.lower() in keymap):
                            if self.mode == 'collect':
                                self.handle_key_input(char)            
                        elif char == 'q':
                            self.get_logger().info('Quit command received')
                            self.running = False
                            break
                        elif char == 'h':
                            self.print_help()
                        elif char == 's':
                            self.print_status()
                        else:
                            # Only show message for printable characters
                            if char.isprintable() and char != ' ':
                                self.get_logger().info(f'Unknown command: "{char}". Press "h" for help.')
                        
            except KeyboardInterrupt:
                self.get_logger().info('KeyboardInterrupt received')
                self.running = False
                break
            except Exception as e:
                self.get_logger().error(f'Error in keyboard listener: {str(e)}')
                break
        
        self.get_logger().info('Keyboard listener stopped')

    def print_help(self):
        """Print help message"""
        self.get_logger().info('=== HELP ===')
        self.get_logger().info('  s: Show status')
        self.get_logger().info('  h: Show this help')
        self.get_logger().info('  q: Quit program')
        self.get_logger().info('============')

    def print_status(self):
        """Print current status"""
        self.get_logger().info('=== STATUS ===')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(f'Latest scan available: {self.latest_scan is not None}')
        self.get_logger().info(f'Latest image available: {self.latest_image is not None}')
        if self.latest_image is not None:
            h, w = self.latest_image.shape[:2]
            self.get_logger().info(f'Image size: {w}x{h}')
        
        # Count saved images in each directory
        for key in keymap.keys():
            key_dir = os.path.join(self.output_dir, key)
            if os.path.exists(key_dir):
                count = len([f for f in os.listdir(key_dir) if f.endswith('.jpg')])
                self.get_logger().info(f'Images in {key}/ directory: {count}')
        self.get_logger().info('==============')

    def handle_key_input(self, key):
        """Handle keyboard input and execute corresponding action"""
        if key in keymap:
            action = keymap[key]
            self.get_logger().info(f'Command: {key}')
            
            # Save data if image is available (save_data=True for all keys)
            save_data = (self.latest_image is not None)
            
            if not save_data:
                self.get_logger().warn('No image available to save!')
            
            # Call publish_cmd_vel with the save_data flag
            self.publish_cmd_vel(
                linear_x=action['linear_x'],
                angular_z=action['angular_z'],
                save_data=save_data,
                key=key
            )
        elif key.lower() in keymap:
            action = keymap[key]
            self.get_logger().info(f'Command: {key}')
            self.publish_cmd_vel(
                linear_x=action['linear_x'],
                angular_z=action['angular_z']
            )  


    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0, save_data=False, key=None):
        """Publish cmd_vel message and optionally save data"""
        
        # Save data first if requested
        if save_data and key and self.latest_image is not None:
            self.save_data(key)
        
        # Then publish cmd_vel
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_x
        cmd_vel_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        self.get_logger().info(f'Published CMD_VEL - Linear: {linear_x:.2f}, Angular: {angular_z:.2f}')

    def img_preprocess(self, image):
        h, w = image.shape[:2]
        crop_height = int(h * crop_top_ratio)
        return image[crop_height:, :, :]
    
    def save_data(self, key):
        """Save current image data to the appropriate directory"""
        if self.latest_image is None:
            self.get_logger().warn('No image data available to save')
            return
        
        time_now = time.time()
        if time_now - self.last_save_time < self.save_interval:
            self.get_logger().info('Save interval not reached, skipping save')
            return
        
        try:
            # Create timestamp with milliseconds
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Create filename and path
            filename = f'{timestamp}.jpg'
            key_dir = os.path.join(self.output_dir, key)
            filepath = os.path.join(key_dir, filename)
            
            # Save image
            preprocessed_image = self.img_preprocess(self.latest_image)
            success = cv2.imwrite(filepath, preprocessed_image)
            
            if success:
                self.get_logger().info(f'✓ Saved image: {filepath}')
            else:
                self.get_logger().error(f'✗ Failed to save image: {filepath}')
                
        except Exception as e:
            self.get_logger().error(f'Error saving data: {str(e)}')

    def shutdown(self):
        """Clean shutdown"""
        self.get_logger().info('Shutting down data collector...')
        self.running = False
        # Send stop command
        self.publish_cmd_vel(0.0, 0.0, False, None)
        # Restore terminal settings
        self.restore_terminal()
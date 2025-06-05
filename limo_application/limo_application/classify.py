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
import requests

from limo_application.constants import (
    crop_top_ratio,
    save_interval,
    keymap, 
    collect_dir,
    prediction_interval,
    forward_object_distance_threshold,
    smooth_action,
)

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
        self.last_key = None
        self.last_save_time = 0
        
        # Control variables
        self.running = True
        
        # Terminal settings for raw input
        self.old_settings = None
        self.setup_terminal()
        
        # Keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()



        # mode setup
        self.output_dir = collect_dir
        if self.mode == 'collect':
            self.setup_output_directories()
            self.get_logger().info('Data Collector initialized.\r')
        elif self.mode == 'predict':
            self.predict_start = False
            self.predict_timer = self.create_timer(prediction_interval, self.predict_action)
            self.get_logger().info('Predict initialized.\r')
            self.get_logger().info('Press SPACE to start/stop prediction.\r')
        else:
            raise ValueError(f'Invalid mode: {mode}. Use "collect" or "predict".')



    def predict_action(self):
        if not self.predict_start:
            return
        
        if self.get_forward_object_distance():
            self.publish_cmd_vel(0,0)
            return
                         
        predict_result = self.predict()
        if predict_result:
            if smooth_action:
                linear_x = predict_result['mean_linear_x']
                angular_z = predict_result['mean_angular_z']
            else:
                linear_x = predict_result['best_linear_x']
                angular_z = predict_result['best_angular_z']
            self.publish_cmd_vel(
                linear_x=linear_x,
                angular_z=angular_z,
                save_data=False
            )
                
    def get_forward_object_distance(self):
        try:
            if self.latest_scan :
                latest_scan_len = len(self.latest_scan)
                last_scan_section_len = latest_scan_len // 3
                # right_list = self.latest_scan[:last_scan_section_len]
                # left_list = self.latest_scan[2*last_scan_section_len:]
                forward_list = self.latest_scan[last_scan_section_len:2*last_scan_section_len]
                forward_list_no_zeros = sorted([x for x in forward_list if x > 0])
                if len(forward_list_no_zeros) == 0:
                    return True
                elif np.mean(forward_list_no_zeros[:10]) < forward_object_distance_threshold:
                    self.get_logger().info(f'Forward object detected: {np.mean(forward_list_no_zeros[:10]):.2f} m\r')
                    return True
                else:
                    return False
        except Exception as e:
            self.get_logger().error(f'Error getting forward object distance: {str(e)}\r')
            
        return False

    def predict(self):
        """Run prediction on the latest image"""
        if self.latest_image is None:
            self.get_logger().warn('No image available for prediction\r')
            return
        
        try:
            preprocessed_image = self.img_preprocess(self.latest_image)
            rgb_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
            _, encoded_image = cv2.imencode('.jpg', rgb_image)
            files = {"file": ("image.jpg", encoded_image.tobytes(), "image/jpeg")}
            response = requests.post("http://127.0.0.1:5000", files=files)
            if response.status_code != 200:
                self.get_logger().error(f'Error in prediction request: {response.status_code} - {response.text}\r')
                return
            self.get_logger().info(f'------ Prediction Result ------\r')
            result = response.json()
            self.get_logger().info(f'result_names - {result["result_names"]}\r')
            self.get_logger().info(f'result_probs - {[f"{p:.2f}" for p in result["result_probs"]]}\r')
            self.get_logger().info(f'best_index - {result["best_index"]}\r')
            self.get_logger().info(f'best_name - {result["best_name"]}\r')
            self.get_logger().info(f'best_prob - {result["best_prob"]:.2f}\r')

            best_linear_x = keymap[result['best_name']]['linear_x']
            best_angular_z = keymap[result['best_name']]['angular_z']
            mean_linear_x = sum(p * keymap[name]['linear_x'] for name, p in zip(result['result_names'], result['result_probs']))
            mean_angular_z = sum(p * keymap[name]['angular_z'] for name, p in zip(result['result_names'], result['result_probs']))
            
            self.get_logger().info(f'-------------------------------\r')
            return {
                'result': result,
                'best_linear_x': best_linear_x,
                'best_angular_z': best_angular_z,
                'mean_linear_x': mean_linear_x,
                'mean_angular_z': mean_angular_z
            }
            
        except Exception as e:
            self.get_logger().error(f'Error during prediction: {str(e)}\r')

    def setup_terminal(self):
        """Setup terminal for raw input (no enter required)"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            self.get_logger().error(f'Error setting up terminal: {str(e)}\r')
            self.old_settings = None

    def restore_terminal(self):
        """Restore terminal settings"""
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception as e:
                self.get_logger().error(f'Error restoring terminal: {str(e)}\r')

    def setup_output_directories(self):
        """Create output directories for each keymap entry"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.get_logger().info(f'Created main output directory: {self.output_dir}\r')
            
            for key in keymap.keys():
                key_dir = os.path.join(self.output_dir, key)
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)
                    self.get_logger().info(f'Created directory: {key_dir}\r')
        except Exception as e:
            self.get_logger().error(f'Error creating directories: {str(e)}\r')

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
            self.get_logger().error(f'Error processing image: {str(e)}\r')


    def keyboard_listener(self):
        """Listen for keyboard input in a separate thread - instant response"""
        self.get_logger().info('Keyboard listener started. Press keys directly (no Enter needed)...\r')
        
        time_last = 0
        while self.running and rclpy.ok():
            time_now = time.time()
            try:
                # Check if input is available using select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if char:= sys.stdin.read(1):
                        if (char != self.last_key) or ((time_now - time_last) > 0.1):
                            time_last = time_now
                            self.last_key = char
                            self.get_logger().info(f'Key pressed: "{char}"\r')
                        else:
                            continue
                        
                        if char == '\x03':  # Ctrl+C
                            self.get_logger().info('Ctrl+C received\r')
                            self.running = False
                            break
                        
                        if self.mode == 'predict':
                            if char == ' ': # spacebar
                                self.predict_start = not self.predict_start
                                if self.predict_start:
                                    self.get_logger().info('Prediction started\r')
                                else:
                                    self.get_logger().info('Prediction stopped\r')
                        
                        if self.mode == 'collect':
                            if char in keymap :
                                self.handle_key_input(char)
                        
            except KeyboardInterrupt:
                self.get_logger().info('KeyboardInterrupt received\r')
                self.running = False
                break
            except Exception as e:
                self.get_logger().error(f'Error in keyboard listener: {str(e)}\r')
                continue
        
        self.get_logger().info('Keyboard listener stopped\r')


    def handle_key_input(self, key):
        """Handle keyboard input and execute corresponding action"""
        if key in keymap:
            action = keymap[key]
            self.get_logger().info(f'Command: {key}\r')
            
            # Save data if image is available (save_data=True for all keys)
            save_data = (self.latest_image is not None)
            
            if not save_data:
                self.get_logger().warn('No image available to save!\r')
            
            # Call publish_cmd_vel with the save_data flag
            self.publish_cmd_vel(
                linear_x=action['linear_x'],
                angular_z=action['angular_z'],
                save_data=save_data,
                key=key
            )


    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0, save_data=False, key=None):
        """Publish cmd_vel message and optionally save data"""
        
        # Save data first if requested
        if save_data and key and self.latest_image is not None:
            self.save_data(key)
        
        # Then publish cmd_vel
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(linear_x)
        cmd_vel_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        self.get_logger().info(f'Published CMD_VEL - Linear: {linear_x:.2f}, Angular: {angular_z:.2f}\r')

    def img_preprocess(self, image):
        h, w = image.shape[:2]
        crop_height = int(h * crop_top_ratio)
        return image[crop_height:, :, :]
    
    def save_data(self, key):
        """Save current image data to the appropriate directory"""
        if self.latest_image is None:
            self.get_logger().warn('No image data available to save\r')
            return
        
        time_now = time.time()
        if time_now - self.last_save_time < save_interval:
            return
        else:
            self.last_save_time = time_now

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
                self.get_logger().info(f'✓ Saved image: {filepath}\r')
            else:
                self.get_logger().error(f'✗ Failed to save image: {filepath}\r')
                
        except Exception as e:
            self.get_logger().error(f'Error saving data: {str(e)}\r')

    def shutdown(self):
        """Clean shutdown"""
        self.get_logger().info('Shutting down data collector...\r')
        self.running = False
        # Send stop command
        self.publish_cmd_vel(0.0, 0.0, False, None)
        # Restore terminal settings
        self.restore_terminal()
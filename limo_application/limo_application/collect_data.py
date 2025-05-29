#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import threading
import os
import time
from datetime import datetime
import sys
import termios
import tty
import select

keymap = {
    '0': {
        'message': 'Stop',
        'linear_x': 0.0,
        'angular_z': 0.0
    },
    '1': {
        'message': 'Forward',
        'linear_x': 0.5,
        'angular_z': 0.0
    },
    '2': {
        'message': 'Backward',
        'linear_x': -0.5,
        'angular_z': 0.0
    },
    '3': {
        'message': 'Left',
        'linear_x': 0.5,
        'angular_z': 0.5
    },
    '4': {
        'message': 'Right',
        'linear_x': 0.5,
        'angular_z': -0.5
    }
}

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')        
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Output directory setup
        self.output_dir = os.path.expanduser('~/output')
        self.setup_output_directories()
        
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
        
        # Control variables
        self.running = True
        
        # Terminal settings for raw input
        self.old_settings = None
        self.setup_terminal()
        
        # Keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.get_logger().info('Data Collector initialized.')
        self.get_logger().info('Controls: 0=Stop, 1=Forward, 2=Backward, 3=Left, 4=Right')
        self.get_logger().info('Press a number key to control robot (NO ENTER needed).')
        self.get_logger().info('Press "q" to quit, "h" for help, "s" for status.')

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
                    char = sys.stdin.read(1).lower()
                    
                    if char:
                        if char == 'q':
                            self.get_logger().info('Quit command received')
                            self.running = False
                            break
                        elif char == 'h':
                            self.print_help()
                        elif char == 's':
                            self.print_status()
                        elif char in keymap:
                            self.handle_key_input(char)
                        elif char == '\x03':  # Ctrl+C
                            self.get_logger().info('Ctrl+C received')
                            self.running = False
                            break
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
            self.get_logger().info(f'Command: {key} - {action["message"]}')
            
            # Save data if image is available (save_data=True for all keys except '0')
            save_data = (key != '0' and self.latest_image is not None)
            
            if key != '0' and self.latest_image is None:
                self.get_logger().warn('No image available to save!')
            
            # Call publish_cmd_vel with the save_data flag
            self.publish_cmd_vel(
                linear_x=action['linear_x'],
                angular_z=action['angular_z'],
                save_data=save_data,
                key=key
            )

    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0, save_data=False, key='0'):
        """Publish cmd_vel message and optionally save data"""
        
        # Save data first if requested
        if save_data and self.latest_image is not None:
            self.save_data(key)
        
        # Then publish cmd_vel
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_x
        cmd_vel_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        self.get_logger().info(f'Published CMD_VEL - Linear: {linear_x:.2f}, Angular: {angular_z:.2f}')

    def save_data(self, key):
        """Save current image data to the appropriate directory"""
        if self.latest_image is None:
            self.get_logger().warn('No image data available to save')
            return
        
        try:
            # Create timestamp with milliseconds
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Create filename and path
            filename = f'{timestamp}.jpg'
            key_dir = os.path.join(self.output_dir, key)
            filepath = os.path.join(key_dir, filename)
            
            # Save image
            success = cv2.imwrite(filepath, self.latest_image)
            
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
        self.publish_cmd_vel(0.0, 0.0, False, '0')
        # Restore terminal settings
        self.restore_terminal()

def main(args=None):
    rclpy.init(args=args)
    data_collector = None
    
    try:
        data_collector = DataCollector()
        
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(data_collector)
        
        data_collector.get_logger().info('Starting data collection...')
        data_collector.get_logger().info('Press keys directly for instant response!')
        
        # Spin until the running flag is False or KeyboardInterrupt
        while data_collector.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            
    except KeyboardInterrupt:
        if data_collector:
            data_collector.get_logger().info('Data collection stopped by user (Ctrl+C)')
    except Exception as e:
        if data_collector:
            data_collector.get_logger().error(f'Error in data collector: {str(e)}')
        else:
            print(f'Error initializing data collector: {str(e)}')
    finally:
        if data_collector:
            data_collector.shutdown()
            data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
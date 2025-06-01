#!/usr/bin/env python3

import os
import ctypes
os.environ["OMP_NUM_THREADS"] = "1" # OpenMP 스레드를 1개로 제한 (TLS 부담을 줄이기 위해)
ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libgomp.so.1", mode=ctypes.RTLD_GLOBAL) # libgomp.so.1 을 RTLD_GLOBAL 모드로 강제 로드해서 “정적 TLS 블록(static TLS block)” 할당 문제를 우회

import rclpy
from limo_application.classify import Classifier

def main(args=None):
    rclpy.init(args=args)
    classifier = None
    
    try:
        classifier = Classifier(mode='predict')
        
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(classifier)
        
        classifier.get_logger().info('Starting prediction mode...')
        
        # Spin until the running flag is False or KeyboardInterrupt
        while classifier.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            
    except KeyboardInterrupt:
        if classifier:
            classifier.get_logger().info('Prediction stopped by user (Ctrl+C)')
    except Exception as e:
        if classifier:
            classifier.get_logger().error(f'Error in prediction mode: {str(e)}')
        else:
            print(f'Error initializing prediction mode: {str(e)}')
    finally:
        if classifier:
            classifier.shutdown()
            classifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
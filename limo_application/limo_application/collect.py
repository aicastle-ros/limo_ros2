#!/usr/bin/env python3

import rclpy
from limo_application.classify import Classifier

def main(args=None):
    rclpy.init(args=args)
    classifier = None
    
    try:
        classifier = Classifier(mode='collect')
        
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(classifier)
        
        # Spin until the running flag is False or KeyboardInterrupt
        while classifier.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            
    except KeyboardInterrupt:
        if classifier:
            classifier.get_logger().info('Data collection stopped by user (Ctrl+C)\r')
    except Exception as e:
        if classifier:
            classifier.get_logger().error(f'Error in data collector: {str(e)}\r')
        else:
            print(f'Error initializing data collector: {str(e)}\r')
    finally:
        if classifier:
            classifier.shutdown()
            classifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
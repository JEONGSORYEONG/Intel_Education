#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving (ì¶©ë í´ê²° ë²ì )

import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState, ButtonState, RGBState, RGBStates

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()  
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # ROS í¼ë¸ë¦¬ì/ìë¸ì¤í¬ë¼ì´ë² ì´ê¸°í
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.publisher_ = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 10)
        self.led_pub = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 10)
        self.last_led_state = None

        # ìë¹ì¤ ìì±
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_serv
        ice(SetBool, '~/set_running', self.set_running_srv_callback)

        # êµ¬ëì ìì±
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
        
        # YOLO í´ë¼ì´ì¸í¸ ì´ê¸°í
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def param_init(self):
        """ê°ì ë íë¼ë¯¸í° ì´ê¸°í"""
        self.start = False
        self.enter = False
        self.right = True
        self.button_state = False

        ### ì°íì  ì´ë²¤í¸ ì²ë¦¬ ë³€ì (ê°ì ë¨) ###
        self.turn_right_detected = False
        self.turn_right_start_time = None
        self.turn_right_progress = 0.0  # ì ì§ì  ì íì ìí ì§íë¥ 
        self.count_right = 0
        self.count_right_miss = 0

        ### ì§ì§ ì´ë²¤í¸ ì²ë¦¬ ë³€ì (ê°ì ë¨) ###
        self.go_detected = False
        self.go_start_time = None
        self.go_progress = 0.0  # ì ì§ì  ì íì ìí ì§íë¥ 
        self.count_go = 0
        self.count_go = 0

        ##ë©ì¸ ì¶ê°€ ë³€ì###
        self.current_state = "LINE_FOLLOW"
        self.crosswalk_stop = False
        self.last_turn_time = 0.0

        ### í¡ë¨ë³´ë ì´ë²¤í¸ ì²ë¦¬ ë³€ì ###
        self.crosswalk_count = 0
        self.crosswalk_distance = 0
        self.crosswalk_stop_start_time = None
        self.crosswalk_length = 0.4

        ### ì£¼ì°¨ ë° ê¸°í€ ë³€ì ###
        self.park_x = -1
        self.start_park = False
        self.count_park = 0
        self.stop = True
        
        ### ìë ê´€ë ¨ ë³€ì ###
        self.normal_speed = 0.2
        self.slow_down_speed = 0.1
        self.start_slow_down = False
        self.count_slow_down = 0

        ### ì í¸ë± ë° ê¸°í€ ###
        self.traffic_light_status = None
        self.red_loss_count = 0
        
        self.object_sub = None
        self.image_sub = None
        self.objects_info = []
        
        ### ë¼ì¸ ì¶ì¢ ê´€ë ¨ (ê°ì ë¨) ###
        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False
        self.last_lane_x = 130  # ì´ì  ë¼ì¸ ìì¹ ì €ì¥
        self.pid_anti_windup_enabled = True  # Anti-windup íì±í

        self.prev_lane_x = -1

        ### ë¹ë¸ë¡í¹ LED ì ì´ ë³€ì ###
        self.led_thread_running = False
        self.led_stop_event = threading.Event()

    ### ì ì§ì  ì°íì  ì²ë¦¬ ë©ìë ###
    def handle_right_turn_smooth(self):
        """ì ì§ì  ì°íì  ì²ë¦¬ ë¡ì§ (ì¶©ë í´ê²°)"""
        if self.turn_right_start_time is None:
            self.turn_right_start_time = time.time()
            self.set_led_right_async()
            self.get_logger().info('ì ì§ì  ì°íì  ìì')
            # PID ì ì´ê¸° ìí ë³´ì¡´ (Anti-windup)
            if self.pid_anti_windup_enabled:
                self.pid.clear()  # ì ë¶ê¸° ì´ê¸°íë¡ windup ë°©ì§€
            
        # ì°íì  ì§íë¥  ê³ì° (0.0 ~ 1.0)
        elapsed_time = time.time() - self.turn_right_start_time
        self.turn_right_progress = min(elapsed_time / 2.0, 1.0)
        
        return self.turn_right_progress >= 1.0  # ìë£ ì¬ë¶€ ë°í

    def calculate_hybrid_control(self, lane_x, turn_progress):
        """íì´ë¸ë¦¬ë ì ì´ ê³ì° (ì°íì  + ë¼ì¸ ì¶ì¢)"""
        twist = Twist()
        
        # ëì  ê°€ì¤ì¹ ê³ì°
        turn_weight = 0.9 * (1.0 - turn_progress * 0.3)  # ì ì§ì  ê°ì
        line_weight = 0.1 + (turn_progress * 0.3)  # ì ì§ì  ì¦ê°€
        
        # ì°íì  ì ì´ ì±ë¶
        turn_linear = self.normal_speed * (1.0 - 0.2 * turn_progress)
        turn_angular = -0.45 * (1.0 - 0.3 * turn_progress)
        
        # ë¼ì¸ ì¶ì¢ ì ì´ ì±ë¶ (ì§€ìì  ìë°ì´í¸)
        line_linear = self.normal_speed
        line_angular = 0.0
        
        if lane_x >= 0:
            self.last_lane_x = lane_x  # ë§ì§€ë§ ì í¨ ë¼ì¸ ìì¹ ì €ì¥
            if lane_x > 150:
                line_angular = -0.45
            else:
                self.pid.SetPoint = 130
                self.pid.update(lane_x)
                # Anti-windup ì²ë¦¬
                if self.pid_anti_windup_enabled:
                    pid_output = common.set_range(self.pid.output, -0.15, 0.15)
                else:
                    pid_output = common.set_range(self.pid.output, -0.1, 0.1)
                line_angular = pid_output
        else:
            # ë¼ì¸ì´ ê°ì§€ëì§€ ìì ë ì´ì  ê° ì ì§€
            if hasattr(self, 'last_lane_x'):
                self.pid.SetPoint = 130
                self.pid.update(self.last_lane_x)
                line_angular = common.set_range(self.pid.output, -0.1, 0.1)
        
        # íì´ë¸ë¦¬ë ì ì´ í©ì±
        twist.linear.x = (turn_linear * turn_weight) + (line_linear * line_weight)
        twist.angular.z = (turn_angular * turn_weight) + (line_angular * line_weight)
        
        # ê¸°ê³ í€ìë³ ì¡°ì 
        if self.machine_type == 'MentorPi_Acker':
            twist.angular.z = twist.linear.x * math.tan(twist.angular.z) / 0.145
        
        return twist

    ### ê°ì ë í¡ë¨ë³´ë ì²ë¦¬ ë©ìë ###
    def handle_crosswalk(self):
        
        """ê°ì ë í¡ë¨ë³´ë ì²ë¦¬ ë©ì»¤ëì¦"""
        if self.crosswalk_count >= 2:
            self.crosswalk_count = 0
            return
        if self.crosswalk_count == 0:
            self.handle_first_crosswalk()
        elif self.crosswalk_count == 1:
            self.handle_second_crosswalk()

    def handle_first_crosswalk(self):
        """ì²« ë²ì§¸ í¡ë¨ë³´ë ì ì§€ ë¡ì§ (ê°ì ë¨)"""
        if self.crosswalk_stop_start_time is None:
            self.get_logger().info('ì²« ë²ì§¸ í¡ë¨ë³´ë ì ì§€ ìì')
            self.crosswalk_stop_start_time = time.time()
            # ì ì§ì  ê°ì
            self.mecanum_pub.publish(Twist())
            self.stop = True
            self.set_led_color(self.stop)
            # PID ìí ë³´ì¡´
            if self.pid_anti_windup_enabled:
                self.pid.clear()

        # 2ì´ ì ì§€ í ì¬ì¶ë°
        if time.time() - self.crosswalk_stop_start_time >= 2.0:
            self.get_logger().info('ì²« ë²ì§¸ í¡ë¨ë³´ë íµê³¼ ìë£')
            self.crosswalk_count = 1
            self.crosswalk_stop_start_time = None
            self.set_led_color(False)

    def handle_second_crosswalk(self):
        """ë ë²ì§¸ í¡ë¨ë³´ë íµê³¼ ë¡ì§ (ê°ì ë¨)"""
        # ì í¸ë± ìí íì¸
        twist = Twist()
        if self.traffic_light_status:
            area = abs(self.traffic_light_status.box[0] - self.traffic_light_status.box[2]) * abs(self.traffic_light_status.box[1] - self.traffic_light_status.box[3])
            if self.traffic_light_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                self.get_logger().info('ì ì ì í¸ë¡ ì ì§ì  ì ì§€')
                self.mecanum_pub.publish(Twist())
                self.stop = True
            elif self.traffic_light_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                self.get_logger().info('ë¹ì ì í¸ë¡ íµê³¼')
                twist.linear.x = self.slow_down_speed
                self.stop = False
            if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                twist.linear.x = self.slow_down_speed
                if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                    self.start_slow_down = False

        # ìê° ê¸°ë° íµê³¼ íì 
        if not hasattr(self, 'crosswalk_pass_start_time'):
            self.crosswalk_pass_start_time = time.time()
            self.get_logger().info('ë ë²ì§¸ í¡ë¨ë³´ë íµê³¼ ìì')

        twist.linear.x = self.slow_down_speed
        self.mecanum_pub.publish(twist)

        # 3ì´ ëì íµê³¼ ì ì§€
        if time.time() - self.crosswalk_pass_start_time >= 3.0:
            self.get_logger().info('ë ë²ì§¸ í¡ë¨ë³´ë íµê³¼ ìë£')
            self.crosswalk_count = 2
            del self.crosswalk_pass_start_time

    def apply_gradual_stop(self):
        """ì ì§ì  ì ì§€ ì ì©"""
        # def gradual_stop_worker():
        #     for i in range(10):
        #         if not self.is_running:
        #             break
        #         twist = Twist()
        #         self.get_logger().info(f'ì ì§ì  ì ì§€ ì¤ : {i}')
        #         twist.linear.x = self.normal_speed * (1.0 - i / 10.0)
        #         self.mecanum_pub.publish(twist)
        #         time.sleep(0.1)
        self.mecanum_pub.publish(Twist())  # ìì  ì ì§€
        
        # threading.Thread(target=gradual_stop_worker, daemon=True).start()

    ### ë¹ë¸ë¡í¹ LED ì ì´ ë©ìë ###
    def set_led_color(self, is_stop):
        """ê¸°ë³¸ LED ìì ì¤ì  (ë¹ë¸ë¡í¹)"""
        if is_stop:
            color = (255, 0, 0)
            state_str = "RED"
        else:
            color = (0, 255, 0)
            state_str = "GREEN"

        if self.last_led_state != state_str:
            msg = RGBStates()
            msg.states = [
                RGBState(index=1, red=color[0], green=color[1], blue=color[2]),
            ]          
            self.publisher_.publish(msg)
            self.led_pub.publish(msg)
            self.last_led_state = state_str

    def set_led_right_async(self, blink_count=3, blink_interval=0.3):
        """ë¹ë¸ë¡í¹ ì°íì  LED ì ë©¸"""
        def blink_worker():
            try:
                self.led_thread_running = True
                self.led_stop_event.clear()
                
                msg_on = RGBStates()
                msg_on2 = RGBStates()
                msg_on.states = [RGBState(index=2, red=255, green=255, blue=0)]
                msg_on2.states = [RGBState(index=2, red=0, green=0,blue=255)]
                msg_off = RGBStates()
                msg_off.states = [RGBState(index=2, red=0, green=0, blue=0)]
                
                for _ in range(blink_count):
                    if self.led_stop_event.wait(0):  # ë¼ë¸ë¡í¹ ì²´í¬
                        break
                    self.publisher_.publish(msg_on)
                    self.led_pub.publish(msg_on2)
                    if self.led_stop_event.wait(blink_interval):
                        break
                    self.publisher_.publish(msg_off)
                    self.led_pub.publish(msg_off)
                    if self.led_stop_event.wait(blink_interval):
                        break
                        
            except Exception as e:
                self.get_logger().error(f'LED ì ë©¸ ì¤ë¥: {e}')
            finally:
                self.led_thread_running = False
        
        if not self.led_thread_running:
            threading.Thread(target=blink_worker, daemon=True).start()

    def stop_led_blink(self):
        """LED ì ë©¸ ì¤ì§€"""
        self.led_stop_event.set()

    ### ë²í¼ ì½ë°± ###
    def button_callback(self, msg):
        if msg.id == 1:
            self.start = True
            self.button_state = True
            self.stop = False
            self.get_logger().info('ìì ë²í¼ ëë¦¼')
        elif msg.id == 2:
            self.start = False
            self.button_state = False
            self.stop = True
            self.crosswalk_count = 0
            self.stop_led_blink()  # LED ì ë©¸ ì¤ì§€
            self.mecanum_pub.publish(Twist())
            self.get_logger().info('ì ì§€ ë²í¼ ëë¦¼')

    ### ë©ì¸ ì²ë¦¬ ë£¨í (ì¶©ë í´ê²°) ###
    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue
        
            result_image = image.copy()
            h, w = image.shape[:2]
            binary_image = self.lane_detect.get_binary(image)
            twist = Twist()

            self.set_led_color(self.stop)
            # ë²í¼ ìíì ë°ë¥¸ ìì ì¡°ê±´
            if self.start and self.button_state:
                # self.stop = False  
                # Determine current state
                if self.turn_right_detected:
                    self.current_state = "TURN_RIGHT"
                # elif self.go_detected:
                #     self.current_state = "GO"
                elif self.traffic_light_status and self.traffic_light_status.class_name == 'red' and self.crosswalk_distance < 100:
                    self.current_state = "RED_LIGHT_STOP"
                elif self.traffic_light_status and self.traffic_light_status.class_name == 'green':
                    # self.current_state = "GREEN_LIGHT_START"
                    self.stop = False
                    self.set_led_color(self.stop)
                    twist.linear.x = self.normal_speed
                    twist.angular.z = 0.0
                    self.mecanum_pub.publish(twist)                    
                    # self.current_state = "LINE_FOLLOW"
                elif self.park_x > 0 and self.crosswalk_distance > 1500:
                    self.get_logger().info("Parking Start!!")                
                    self.current_state = "PARKING"
                else:
                    self.current_state = "LINE_FOLLOW"

                self.get_logger().info(f'Current State: {self.current_state}')                 


                # Execute based on state
                if self.current_state == "TURN_RIGHT":
                    self.set_led_right_async()
                    twist.linear.x = self.normal_speed
                    twist.angular.z = -0.45
                    self.mecanum_pub.publish(twist)
                    self.get_logger().info("Executing right turn.")
                    time.sleep(2)
                    self.mecanum_pub.publish(Twist())
                    self.turn_right_detected = False
                    self.current_state = "LINE_FOLLOW"
                    continue


                elif self.current_state == "GO":
                    self.mecanum_pub.publish(Twist())
                    self.stop = True
                    time.sleep(2)
                    self.go_detected = False
                    self.current_state = "LINE_FOLLOW"
                    continue

                elif self.current_state == "CROSSWALK_STOP":
                    self.get_logger().info("Crosswalk detected: Stopping")
                    # self.crosswalk_stop = True
                    # self.mecanum_pub.publish(Twist())
                    # self.stop = True
                    # time.sleep(2)
                    #self.handle_crosswalk()  
                    continue

                elif self.current_state == "RED_LIGHT_STOP":
                    self.get_logger().info("Red light detected: Stopping")
                    self.mecanum_pub.publish(Twist())
                    self.stop = True
                    self.set_led_color(self.stop)  # ì ì§€ ì ë¹¨ê°ë¶ ì¼ê¸°
                    continue

                # elif self.current_state == "GREEN_LIGHT_START":
                #     self.get_logger().info("Green light detected: Starting")
                #     self.stop = False
                #     twist.linear.x = self.normal_speed
                #     twist.angular.z = 0.0
                #     self.mecanum_pub.publish(twist)                    
                #     # self.current_state = "LINE_FOLLOW"
                #     continue

                elif self.current_state == "PARKING":
                    # twist.linear.x = self.slow_down_speed
                    self.count_park += 1
                    self.get_logger().info(f'{self.count_park}')
                    if self.count_park == 20:
                        # self.mecanum_pub.publish(Twist())
                        self.start_park = True                         
                        self.stop = True
                        self.count_park = 0
                        threading.Thread(target=self.park_action).start()
                        
                    else:
                        self.mecanum_pub.publish(twist)
                    continue

                elif self.current_state == "LINE_FOLLOW":
                    result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                    if lane_x >= 0 and not self.stop:
                        if lane_x > 150:
                            self.count_turn += 1
                            if self.count_turn > 5 and not self.start_turn:
                                self.set_led_right_async()
                                self.start_turn = True
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            if self.machine_type != 'MentorPi_Acker':
                                twist.angular.z = -0.45  # turning speed
                            else:
                                twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                        else:
                            self.stop = False
                            self.count_turn = 0
                            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                                self.start_turn = False
                            if not self.start_turn:
                                self.pid.SetPoint = 130
                                self.pid.update(lane_x)
                                if self.machine_type != 'MentorPi_Acker':
                                    twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                            else:
                                if self.machine_type == 'MentorPi_Acker':
                                    twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                        
                        self.mecanum_pub.publish(twist)
                    else:
                        self.pid.clear()

            
            self.set_led_color(self.stop) 

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


    ### ê°ì²´ ê²€ì¶ ì½ë°± (ê¸°ì¡´ê³¼ ëì¼) ###
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            max_y = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                if class_name == 'crosswalk':
                    if 100 < self.crosswalk_distance and not self.start_slow_down:  # The robot starts to slow down only when it is close enough to the zebra crossing
                        self.count_crosswalk += 1
                        if self.count_crosswalk == 5:  # judge multiple times to prevent false detection
                            self.count_crosswalk = 0
                            self.crosswalk_count += 1 
                            self.start_slow_down = True  # sign for slowing down
                            self.count_slow_down = time.time()  # fixing time for slowing down
                    else:  # need to detect continuously, otherwise reset
                        self.count_crosswalk = 0
                    if center[1] > max_y:
                        max_y = center[1]
                elif class_name == 'right':
                    self.count_right += 1
                    if self.count_right >= 5:
                        self.turn_right_detected = True
                        self.count_right = 0
                elif class_name == 'park':
                    self.park_x = center[0]

                elif class_name == 'go':
                    self.count_go += 1
                    if self.count_go >= 5:
                        self.go_detected = True
                        self.count_go = 0

                elif class_name in ['red', 'green']:
                    self.traffic_light_status = i
                
            self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
            
            self.crosswalk_distance = max_y

    ### ê¸°ì¡´ ë©ìëë¤ ì ì§€ ###
    def init_process(self):
        self.timer.cancel()
        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "ìì¨ì£¼í ì§ì")
        with self.lock:
            self.start = False
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "ìì¨ì£¼í ì¢ë£")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "ì¤í ìí ì¤ì ")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):
        self.is_running = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)
    
    def park_action(self):
        """ì£¼ì°¨ ëì (ê°ì ë¨)"""
        try:
            if self.machine_type == 'MentorPi_Mecanum':
                self.get_logger().info('111111')
                self.get_logger().info('ì¤ë¥¸ìª½(ì°ì¸¡) íí¹ ìì')
                duration = 0.38 / 0.2  # 3ì´
                start_time = time.time()
                twist = Twist()
                twist.linear.y = -0.2

                while time.time() - start_time < duration:
                    self.mecanum_pub.publish(twist)
                    time.sleep(0.1)  # 10Hz ì£¼ê¸°ë¡ publish

                self.mecanum_pub.publish(Twist())
                #time.sleep(1.5 / 0.5)  # ì´ë ìê° ì¡°ì .

            # elif self.machine_type == 'MentorPi_Acker':
            #     self.get_logger().info('2222222')
            #     twist = Twist()
            #     twist.linear.x = 0.15
            #     twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(3)

            #     twist = Twist()
            #     twist.linear.x = 0.15
            #     twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(2)

            #     twist = Twist()
            #     twist.linear.x = -0.15
            #     twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(1.5)
            # else:
            #     self.get_logger().info('33333')
            #     twist = Twist()
            #     twist.angular.z = -1
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(1.5)
            #     self.mecanum_pub.publish(Twist())
            #     twist = Twist()
            #     twist.linear.x = 0.2
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(0.65/0.2)
            #     self.mecanum_pub.publish(Twist())
            #     twist = Twist()
            #     twist.angular.z = 1
            #     self.mecanum_pub.publish(twist)
            #     time.sleep(1.5)
        except Exception as e:
            self.get_logger().error(f'ì£¼ì°¨ ëì ì¤ë¥: {e}')
        finally:
            self.get_logger().info('park_action í¨ì ì¢ë£')
            self.mecanum_pub.publish(Twist())  # í­ì ì ì§€ë¡ ë§ë¬´ë¦¬
            # self.count_park = 0
            self.start_park = False
            self.stop = False
            

def main():
    # rclpy.init()
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

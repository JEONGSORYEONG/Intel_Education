#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving (충돌 해결 버전)

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

        # ROS 퍼블리셔/서브스크라이버 초기화
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.publisher_ = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 10)
        self.last_led_state = None

        # 서비스 생성
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        # 구독자 생성
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
        
        # YOLO 클라이언트 초기화
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def param_init(self):
        """개선된 파라미터 초기화"""
        self.start = False
        self.enter = False
        self.right = True
        self.button_state = False

        ### 우회전 이벤트 처리 변수 (개선됨) ###
        self.turn_right_detected = False
        self.turn_right_start_time = None
        self.turn_right_progress = 0.0  # 점진적 전환을 위한 진행률
        self.count_right = 0
        self.count_right_miss = 0

        ### 횡단보도 이벤트 처리 변수 ###
        self.crosswalk_count = 0
        self.crosswalk_distance = 0
        self.crosswalk_stop_start_time = None
        self.crosswalk_length = 0.4

        ### 주차 및 기타 변수 ###
        self.park_x = -1
        self.start_park = False
        self.count_park = 0
        self.stop = True
        
        ### 속도 관련 변수 ###
        self.normal_speed = 0.2
        self.slow_down_speed = 0.1
        self.start_slow_down = False
        self.count_slow_down = 0

        ### 신호등 및 기타 ###
        self.traffic_light_status = None
        self.red_loss_count = 0
        
        self.object_sub = None
        self.image_sub = None
        self.objects_info = []
        
        ### 라인 추종 관련 (개선됨) ###
        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False
        self.last_lane_x = 130  # 이전 라인 위치 저장
        self.pid_anti_windup_enabled = True  # Anti-windup 활성화

        self.prev_lane_x = -1

        ### 비블로킹 LED 제어 변수 ###
        self.led_thread_running = False
        self.led_stop_event = threading.Event()

    ### 점진적 우회전 처리 메서드 ###
    def handle_right_turn_smooth(self):
        """점진적 우회전 처리 로직 (충돌 해결)"""
        if self.turn_right_start_time is None:
            self.turn_right_start_time = time.time()
            self.set_led_right_async()
            self.get_logger().info('점진적 우회전 시작')
            # PID 제어기 상태 보존 (Anti-windup)
            if self.pid_anti_windup_enabled:
                self.pid.clear()  # 적분기 초기화로 windup 방지
            
        # 우회전 진행률 계산 (0.0 ~ 1.0)
        elapsed_time = time.time() - self.turn_right_start_time
        self.turn_right_progress = min(elapsed_time / 2.0, 1.0)
        
        return self.turn_right_progress >= 1.0  # 완료 여부 반환

    def calculate_hybrid_control(self, lane_x, turn_progress):
        """하이브리드 제어 계산 (우회전 + 라인 추종)"""
        twist = Twist()
        
        # 동적 가중치 계산
        turn_weight = 0.9 * (1.0 - turn_progress * 0.3)  # 점진적 감소
        line_weight = 0.1 + (turn_progress * 0.3)  # 점진적 증가
        
        # 우회전 제어 성분
        turn_linear = self.normal_speed * (1.0 - 0.2 * turn_progress)
        turn_angular = -0.45 * (1.0 - 0.3 * turn_progress)
        
        # 라인 추종 제어 성분 (지속적 업데이트)
        line_linear = self.normal_speed
        line_angular = 0.0
        
        if lane_x >= 0:
            self.last_lane_x = lane_x  # 마지막 유효 라인 위치 저장
            if lane_x > 150:
                line_angular = -0.45
            else:
                self.pid.SetPoint = 130
                self.pid.update(lane_x)
                # Anti-windup 처리
                if self.pid_anti_windup_enabled:
                    pid_output = common.set_range(self.pid.output, -0.15, 0.15)
                else:
                    pid_output = common.set_range(self.pid.output, -0.1, 0.1)
                line_angular = pid_output
        else:
            # 라인이 감지되지 않을 때 이전 값 유지
            if hasattr(self, 'last_lane_x'):
                self.pid.SetPoint = 130
                self.pid.update(self.last_lane_x)
                line_angular = common.set_range(self.pid.output, -0.1, 0.1)
        
        # 하이브리드 제어 합성
        twist.linear.x = (turn_linear * turn_weight) + (line_linear * line_weight)
        twist.angular.z = (turn_angular * turn_weight) + (line_angular * line_weight)
        
        # 기계 타입별 조정
        if self.machine_type == 'MentorPi_Acker':
            twist.angular.z = twist.linear.x * math.tan(twist.angular.z) / 0.145
        
        return twist

    ### 개선된 횡단보도 처리 메서드 ###
    def handle_crosswalk(self):
        
        """개선된 횡단보도 처리 메커니즘"""
        if self.crosswalk_count >= 2:
            self.crosswalk_count = 0
            return
        if self.crosswalk_count == 0:
            self.handle_first_crosswalk()
        elif self.crosswalk_count == 1:
            self.handle_second_crosswalk()

    def handle_first_crosswalk(self):
        """첫 번째 횡단보도 정지 로직 (개선됨)"""
        if self.crosswalk_stop_start_time is None:
            self.get_logger().info('첫 번째 횡단보도 정지 시작')
            self.crosswalk_stop_start_time = time.time()
            # 점진적 감속
            self.mecanum_pub.publish(Twist())
            self.stop = True
            self.set_led_color(self.stop)
            # PID 상태 보존
            if self.pid_anti_windup_enabled:
                self.pid.clear()

        # 2초 정지 후 재출발
        if time.time() - self.crosswalk_stop_start_time >= 2.0:
            self.get_logger().info('첫 번째 횡단보도 통과 완료')
            self.crosswalk_count = 1
            self.crosswalk_stop_start_time = None
            self.set_led_color(False)

    def handle_second_crosswalk(self):
        """두 번째 횡단보도 통과 로직 (개선됨)"""
        # 신호등 상태 확인
        twist = Twist()
        if self.traffic_light_status:
            area = abs(self.traffic_light_status.box[0] - self.traffic_light_status.box[2]) * abs(self.traffic_light_status.box[1] - self.traffic_light_status.box[3])
            if self.traffic_light_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                self.get_logger().info('적색 신호로 점진적 정지')
                self.mecanum_pub.publish(Twist())
                self.stop = True
            elif self.traffic_light_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                self.get_logger().info('녹색 신호로 통과')
                twist.linear.x = self.slow_down_speed
                self.stop = False
            if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                twist.linear.x = self.slow_down_speed
                if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                    self.start_slow_down = False

        # 시간 기반 통과 판정
        if not hasattr(self, 'crosswalk_pass_start_time'):
            self.crosswalk_pass_start_time = time.time()
            self.get_logger().info('두 번째 횡단보도 통과 시작')

        twist.linear.x = self.slow_down_speed
        self.mecanum_pub.publish(twist)

        # 3초 동안 통과 유지
        if time.time() - self.crosswalk_pass_start_time >= 3.0:
            self.get_logger().info('두 번째 횡단보도 통과 완료')
            self.crosswalk_count = 2
            del self.crosswalk_pass_start_time

    def apply_gradual_stop(self):
        """점진적 정지 적용"""
        # def gradual_stop_worker():
        #     for i in range(10):
        #         if not self.is_running:
        #             break
        #         twist = Twist()
        #         self.get_logger().info(f'점진적 정지 중 : {i}')
        #         twist.linear.x = self.normal_speed * (1.0 - i / 10.0)
        #         self.mecanum_pub.publish(twist)
        #         time.sleep(0.1)
        self.mecanum_pub.publish(Twist())  # 완전 정지
        
        # threading.Thread(target=gradual_stop_worker, daemon=True).start()

    ### 비블로킹 LED 제어 메서드 ###
    def set_led_color(self, is_stop):
        """기본 LED 색상 설정 (비블로킹)"""
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
            self.last_led_state = state_str

    def set_led_right_async(self, blink_count=3, blink_interval=0.3):
        """비블로킹 우회전 LED 점멸"""
        def blink_worker():
            try:
                self.led_thread_running = True
                self.led_stop_event.clear()
                
                msg_on = RGBStates()
                msg_on.states = [RGBState(index=2, red=255, green=255, blue=0)]
                msg_off = RGBStates()
                msg_off.states = [RGBState(index=2, red=0, green=0, blue=0)]
                
                for _ in range(blink_count):
                    if self.led_stop_event.wait(0):  # 논블로킹 체크
                        break
                    self.publisher_.publish(msg_on)
                    if self.led_stop_event.wait(blink_interval):
                        break
                    self.publisher_.publish(msg_off)
                    if self.led_stop_event.wait(blink_interval):
                        break
                        
            except Exception as e:
                self.get_logger().error(f'LED 점멸 오류: {e}')
            finally:
                self.led_thread_running = False
        
        if not self.led_thread_running:
            threading.Thread(target=blink_worker, daemon=True).start()

    def stop_led_blink(self):
        """LED 점멸 중지"""
        self.led_stop_event.set()

    ### 버튼 콜백 ###
    def button_callback(self, msg):
        if msg.id == 1:
            self.start = True
            self.button_state = True
            self.stop = False
            self.get_logger().info('시작 버튼 눌림')
        elif msg.id == 2:
            self.start = False
            self.button_state = False
            self.stop = True
            self.crosswalk_count = 0
            self.stop_led_blink()  # LED 점멸 중지
            self.mecanum_pub.publish(Twist())
            self.get_logger().info('정지 버튼 눌림')

    ### 메인 처리 루프 (충돌 해결) ###
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
            if self.start and self.button_state: 
                h, w = image.shape[:2]
                binary_image = self.lane_detect.get_binary(image)
                twist = Twist()
                # 항상 라인 검출 수행 (하이브리드 제어 지원)
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                
                #횡단보드 
                if self.crosswalk_distance > 70 and self.crosswalk_count < 2:
                    self.get_logger().info('횡단보드 카운트: {}'.format(self.crosswalk_count))
                    self.handle_crosswalk()

                ## 우회전 간판 확인 ###
                elif self.turn_right_detected:
                    # 점진적 우회전 처리
                    turn_complete = self.handle_right_turn_smooth()
                    # 하이브리드 제어 적용
                    twist = self.calculate_hybrid_control(lane_x, self.turn_right_progress)
                    self.mecanum_pub.publish(twist)
                    
                    if turn_complete:
                        self.turn_right_detected = False
                        self.turn_right_start_time = None
                        self.turn_right_progress = 0.0
                        self.stop_led_blink()
                        self.get_logger().info('점진적 우회전 완료')
                #차선 감지
                else:
                    # 기본 주행 로직 (개선됨)
                    if lane_x >= 0 and not self.stop :
                        twist = Twist()
                        self.set_led_color(self.stop)
                        if lane_x > 150:
                            self.count_turn += 1
                            if self.count_turn > 5 and not self.start_turn:
                                self.start_turn = True
                                self.set_led_right_async()
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            if self.machine_type != 'MentorPi_Acker':
                                twist.angular.z = -0.45
                            else:
                                twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                        else:
                            self.count_turn = 0
                            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                                self.start_turn = False
                            if not self.start_turn:
                                self.pid.SetPoint = 130
                                self.pid.update(lane_x)

                                # Anti-windup PID 적용
                                if self.pid_anti_windup_enabled:
                                    pid_output = common.set_range(self.pid.output, -0.15, 0.15)
                                else:
                                    pid_output = common.set_range(self.pid.output, -0.1, 0.1)
                                
                                if self.machine_type != 'MentorPi_Acker':
                                    twist.angular.z = pid_output
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(pid_output) / 0.145
                            else:
                                if self.machine_type == 'MentorPi_Acker':
                                    twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                        
                        twist.linear.x = self.normal_speed
                        self.mecanum_pub.publish(twist)
                        # 마지막 유효 라인 위치 저장
                        self.last_lane_x = lane_x
                    else:
                        # PID 적분기 초기화 (Anti-windup)
                        if self.pid_anti_windup_enabled:
                            self.pid.clear()

                # 주차 처리 (기존 로직 유지)
                if 0 < self.park_x and not self.start_park:
                    self.count_park += 1  
                    self.get_logger().info('주차 시도 중: {}'.format(self.count_park))
                    if self.count_park >= 15:
                        # self.apply_gradual_stop()
                        self.start_park = True
                        self.get_logger().info('주차 시작')
                        self.stop = True
                        threading.Thread(target=self.park_action).start()
                    # self.mecanum_pub.publish(twist)
                elif self.park_x <= 0:
                    # 'park' 표지판이 사라지면 플래그와 카운트 초기화
                    self.start_park = False
                    self.count_park = 0

                # 객체 검출 결과 시각화
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )
            else:
                time.sleep(0.01)
                 
            # LED 상태 업데이트 및 이미지 발행
            self.set_led_color(self.stop)
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
            
            # 프레임 레이트 제어
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
                
        rclpy.shutdown()

    ### 객체 검출 콜백 (기존과 동일) ###
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if not self.objects_info:
            self.traffic_light_status = None
            self.crosswalk_distance = 0
            return
        max_y = 0
        self.park_x = -1  # park_x 초기화
        for i in self.objects_info:
            class_name = i.class_name
            center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
            
            if class_name == 'crosswalk':  
                if center[1] > max_y:
                    max_y = center[1]
            elif class_name == 'right':
                self.count_right += 1
                self.count_right_miss = 0
                if self.count_right >= 5:
                    self.turn_right_detected = True
                    self.count_right = 0
            elif class_name == 'park':
                self.park_x = center[0]
            elif class_name in ['red', 'green']:
                self.traffic_light_status = i
               
        self.crosswalk_distance = max_y

    ### 기존 메서드들 유지 ###
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
        self.get_logger().info('\033[1;32m%s\033[0m' % "자율주행 진입")
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
        self.get_logger().info('\033[1;32m%s\033[0m' % "자율주행 종료")
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
        self.get_logger().info('\033[1;32m%s\033[0m' % "실행 상태 설정")
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
        """주차 동작 (개선됨)"""
        try:
            if self.machine_type == 'MentorPi_Mecanum':
                self.get_logger().info('111111')
                self.get_logger().info('오른쪽(우측) 파킹 시작')
                duration = 0.38 / 0.2  # 3초
                start_time = time.time()
                twist = Twist()
                twist.linear.y = -0.2

                while time.time() - start_time < duration:
                    self.mecanum_pub.publish(twist)
                    time.sleep(0.1)  # 10Hz 주기로 publish

                self.mecanum_pub.publish(Twist())
                #time.sleep(1.5 / 0.5)  # 이동 시간 조정.

            elif self.machine_type == 'MentorPi_Acker':
                self.get_logger().info('2222222')
                twist = Twist()
                twist.linear.x = 0.15
                twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
                self.mecanum_pub.publish(twist)
                time.sleep(3)

                twist = Twist()
                twist.linear.x = 0.15
                twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
                self.mecanum_pub.publish(twist)
                time.sleep(2)

                twist = Twist()
                twist.linear.x = -0.15
                twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
                self.mecanum_pub.publish(twist)
                time.sleep(1.5)
            else:
                self.get_logger().info('33333')
                twist = Twist()
                twist.angular.z = -1
                self.mecanum_pub.publish(twist)
                time.sleep(1.5)
                self.mecanum_pub.publish(Twist())
                twist = Twist()
                twist.linear.x = 0.2
                self.mecanum_pub.publish(twist)
                time.sleep(0.65/0.2)
                self.mecanum_pub.publish(Twist())
                twist = Twist()
                twist.angular.z = 1
                self.mecanum_pub.publish(twist)
                time.sleep(1.5)
        except Exception as e:
            self.get_logger().error(f'주차 동작 오류: {e}')
        finally:
            self.get_logger().info('park_action 함수 종료')
            self.mecanum_pub.publish(Twist())  # 항상 정지로 마무리
            self.stop = False
            

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

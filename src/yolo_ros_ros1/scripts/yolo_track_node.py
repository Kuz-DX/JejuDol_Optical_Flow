#!/usr/bin/env python
import rclpy
import os
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
from ultralytics import YOLO

# Import custom messages
from yolo_msgs.msg import BoundingBox2D, Detection, DetectionArray

class DynamicObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_track_node")

        # Parameters
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("imgsz_height", 360)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("bbox_area_threshold", 4500)
        self.declare_parameter("motion_threshold_px", 10.0)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_model_path = os.path.join(script_dir, 'best.pt')
        self.declare_parameter("model_path", default_model_path)

        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.bbox_area_threshold = self.get_parameter("bbox_area_threshold").get_parameter_value().integer_value
        self.MOTION_THRESHOLD_PX = self.get_parameter("motion_threshold_px").get_parameter_value().double_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        
        self.steering_angle = 0.0

        self.get_logger().info(f"loading YOLO Model: {model_path}")
        self.model = YOLO("yolov8n.pt")

        # Publishers
        self._tracking_pub = self.create_publisher(DetectionArray, "tracking", 10)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._info_pub = self.create_publisher(String, "cone_info", 10)
        self._motion_error_pub = self.create_publisher(String, "motion_error_info", 10)
        self._dynamic_obstacle_pub = self.create_publisher(Bool, "/dynamic_obstacle", 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, 1)
        self.steering_sub = self.create_subscription(Float32, "/steering_angle", self.steering_angle_cb, 10)
        self.get_logger().info(f"image topic subscription start: {self.image_topic}")

        self.cv_bridge = CvBridge()

        self.old_gray = None
        self.prev_bg_features = None
        self.tracked_cones = {}
        
        self.lk_params = dict(winSize=(31, 31), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

        self.get_logger().info("Node setting complete")
    
    def steering_angle_cb(self, msg):
        self.steering_angle = msg.data

    def image_cb(self, msg: Image):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"image translate fail: {e}")
            return

        cv_image = cv2.resize(cv_image, (self.imgsz_width, self.imgsz_height))
        height, width, _ = cv_image.shape
        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        debug_image = cv_image.copy()

        try:
            results = self.model.track(cv_image, persist=True, conf=self.threshold, iou=self.iou)[0]
        except Exception as e:
            self.get_logger().error(f"YOLO track error: {e}")
            return

        # 1. 배경 마스크 생성 (YOLO로 검출된 콘 영역을 제외)
        background_mask = np.ones(frame_gray.shape, dtype=np.uint8) * 255
        if results.boxes:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                cv2.rectangle(background_mask, (x1, y1), (x2, y2), 0, -1)

        # 2. 배경 마스크를 사용하여 배경 특징점 추출
        ego_motion_matrix = None
        if self.old_gray is not None and self.prev_bg_features is not None and len(self.prev_bg_features) > 4:
            # Optical Flow 계산
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.prev_bg_features, None, **self.lk_params)
            
            # 추적에 성공한 특징점만 필터링
            good_old = self.prev_bg_features[st == 1]
            good_new = p1[st == 1]

            if len(good_new) > 4:
                # Homography 행렬을 계산하여 ego-motion 보상/ 카메라 움직임을 모델링
                ego_motion_matrix, h_mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)

        tracked_detections = DetectionArray()
        tracked_detections.header = msg.header
        cone_info_list = []
        current_tracked_cones = {}
        is_dynamic_obstacle_detected = False

        # 3. 추적된 각 콘에 대해 동적/정적 판단 및 시각화
        if results.boxes and results.boxes.id is not None:
            boxes_xywh = results.boxes.xywh.cpu()
            confs = results.boxes.conf.cpu()
            clss = results.boxes.cls.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                x_center, y_center, w, h = boxes_xywh[i].numpy().astype(int)
                class_idx = int(clss[i].numpy())
                yolo_label = results.names[class_idx]
                
                is_dynamic = False
                motion_error = 0.0

                if ego_motion_matrix is not None and track_id in self.tracked_cones:
                    bbox_area = w * h
                    is_in_central_roi = (width // 3 < x_center < 2 * width // 3)

                    if bbox_area > self.bbox_area_threshold and is_in_central_roi:
                        prev_pos = np.array([[self.tracked_cones[track_id]['position']]], dtype=np.float32)
                        predicted_pos = cv2.perspectiveTransform(prev_pos, ego_motion_matrix)[0][0]
                        actual_pos = np.array([x_center, y_center])
                        motion_error = np.linalg.norm(actual_pos - predicted_pos)

                        motion_error_info_msg = String()
                        motion_error_info_msg.data = f"ID {track_id}: Motion Error {motion_error:.2f} px"
                        self._motion_error_pub.publish(motion_error_info_msg)

                        cv2.circle(debug_image, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (0, 0, 255), -1)
                        cv2.line(debug_image, (int(predicted_pos[0]), int(predicted_pos[1])), (int(actual_pos[0]), int(actual_pos[1])), (0, 255, 255), 2)
                        
                        if motion_error > self.MOTION_THRESHOLD_PX and -15 < self.steering_angle < 15 and is_in_central_roi:
                            is_dynamic = True
                            is_dynamic_obstacle_detected = True
                            self.get_logger().warn(f'DYNAMIC OBSTACLE! ID: {track_id}, Motion Error: {motion_error:.2f} px')

                final_label = yolo_label
                if is_dynamic:
                    final_label += " (Dynamic)"

                current_tracked_cones[track_id] = {'position': (x_center, y_center), 'class_name': yolo_label}

                detection = Detection()
                detection.id = str(track_id)
                detection.class_id = class_idx
                detection.class_name = final_label
                detection.score = float(confs[i].numpy())
                
                bbox = BoundingBox2D()
                bbox.center.position.x = float(x_center)
                bbox.center.position.y = float(y_center)
                bbox.size.x = float(w)
                bbox.size.y = float(h)
                detection.bbox = bbox
                tracked_detections.detections.append(detection)

                cone_info_list.append(f"ID {track_id}: {final_label} ({x_center}, {y_center})")

                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                color = (0, 255, 255) if is_dynamic else (0, 0, 0)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                
                bbox_area = w * h
                if is_dynamic:
                    cv2.putText(debug_image, f"{motion_error:.1f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(debug_image, f"{bbox_area}", (x1, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 10)
                else:
                    cv2.putText(debug_image, f"{bbox_area}", (x1, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.old_gray = frame_gray.copy()
        self.prev_bg_features = cv2.goodFeaturesToTrack(self.old_gray, mask=background_mask, **self.feature_params)
        self.tracked_cones = current_tracked_cones

        if is_dynamic_obstacle_detected:
            cv2.putText(debug_image, "Dynamic", (200, self.imgsz_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 0, 255), 5)

        self._tracking_pub.publish(tracked_detections)
        info_msg = String()
        info_msg.data = "; ".join(cone_info_list)
        self._info_pub.publish(info_msg)

        dynamic_obstacle_msg = Bool()
        dynamic_obstacle_msg.data = is_dynamic_obstacle_detected
        self._dynamic_obstacle_pub.publish(dynamic_obstacle_msg)

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self._dbg_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

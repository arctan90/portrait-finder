import cv2
import mediapipe as mp
import numpy as np
import os
import folder_paths

# 定义支持的视频格式
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'webm']

class VideoFrontalDetectorNode:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取 input 目录中的视频文件
        input_dir = folder_paths.get_input_directory()
        video_files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                ext = f.split('.')[-1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_files.append(f)
                    
        return {
            "required": {
                "video": (sorted(video_files),),  # 从 input 目录选择视频
                "confidence_threshold": ("FLOAT", {
                    "default": 80.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "frame_skip": ("INT", {  # 添加帧跳过选项
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "video/frontal"

    def is_frontal_pose(self, results):
        if not results.pose_landmarks:
            return 0.0
        
        landmarks = results.pose_landmarks.landmark
        
        # 检查肩膀是否平行于相机
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_diff = abs(left_shoulder.z - right_shoulder.z)
        
        # 检查躯干是否正对相机
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        mid_hip = landmarks[self.mp_pose.PoseLandmark.MID_HIP]
        vertical_alignment = abs(nose.x - mid_hip.x)
        
        # 计算总体置信度
        confidence = (1.0 - shoulder_diff) * (1.0 - vertical_alignment) * 100
        return confidence

    def is_frontal_face(self, results):
        if not results.detections:
            return 0.0
        
        # 获取第一个检测到的人脸
        detection = results.detections[0]
        
        # 获取人脸框的关键点
        bbox = detection.location_data.relative_bounding_box
        
        # 计算人脸框的宽高比
        aspect_ratio = bbox.width / bbox.height
        
        # 根据宽高比和置信度计算正脸程度
        confidence = detection.score[0] * (1.0 - abs(aspect_ratio - 0.75)) * 100
        return confidence

    def process(self, video, confidence_threshold, frame_skip):
        try:
            # 构建完整的视频路径
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.isfile(video_path):
                raise ValueError(f"找不到视频文件: {video}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video}")

            best_frame = None
            best_confidence = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 按 frame_skip 跳过帧
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测姿态和人脸
                pose_results = self.pose.process(frame_rgb)
                face_results = self.face_detection.process(frame_rgb)
                
                # 计算置信度
                pose_confidence = self.is_frontal_pose(pose_results)
                face_confidence = self.is_frontal_face(face_results)
                
                # 综合置信度
                total_confidence = min(pose_confidence, face_confidence)
                
                # 更新最佳帧
                if total_confidence > best_confidence:
                    best_confidence = total_confidence
                    best_frame = frame_rgb
                    
                # 如果达到阈值则提前结束
                if best_confidence >= confidence_threshold:
                    break
            
            cap.release()
            
            if best_frame is None:
                raise ValueError("未能找到符合条件的帧")
                
            return (best_frame,)
            
        except Exception as e:
            raise ValueError(f"处理视频时出错: {str(e)}")
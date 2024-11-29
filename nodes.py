import cv2
import mediapipe as mp
import numpy as np
import os
import folder_paths
import hashlib
import torch  # 添加这行导入

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
            min_detection_confidence=0.3,
            model_selection=1
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
                "test_first_frame": ("BOOLEAN", {  # 添加测试开关
                    "default": True,
                }),
            },
        }
    
    # 添加 WIDGETS 定义
    @classmethod
    def WIDGETS(cls):
        return {
            "video": ("videoplayer", {"format": VIDEO_EXTENSIONS}),
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
        # 使用左右臀部的中点代替 MID_HIP
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        
        # 计算垂直对齐度
        vertical_alignment = abs(nose.x - mid_hip_x)
        
        # 计算总体置信度
        confidence = (1.0 - shoulder_diff) * (1.0 - vertical_alignment) * 100
        return confidence

    def is_frontal_face(self, results):
        if not results.detections:
            return 0.0
        
        # 获取所有检测到的人脸中置信度最高的
        best_detection = max(results.detections, key=lambda x: x.score[0])
        
        # 获取人脸框的关键点
        bbox = best_detection.location_data.relative_bounding_box
        
        # 计算人脸框的宽高比
        aspect_ratio = bbox.width / bbox.height  # 可能需要调整为 height/width
        ideal_ratio = 0.8  # 理想宽高比
        ratio_tolerance = 0.8  # 增大容差范围
        
        # 计算宽高比的得分（使用更宽松的评分方式）
        ratio_diff = abs(aspect_ratio - ideal_ratio) / ideal_ratio
        ratio_score = max(0, 1.0 - (ratio_diff / ratio_tolerance))
        
        # 调整人脸大小得分计算
        face_area = bbox.width * bbox.height
        min_face_area = 0.01  # 最小可接受的人脸区域（相对于图像）
        face_size_score = min(face_area / min_face_area, 1.0)
        
        # 计算最终置信度（调整各项权重）
        detection_weight = 0.5    # 检测置信度权重
        ratio_weight = 0.3       # 宽高比权重
        size_weight = 0.2        # 大小权重
        
        confidence = (
            (best_detection.score[0] * detection_weight +  # 检测置信度
             ratio_score * ratio_weight +                 # 宽高比得分
             face_size_score * size_weight) *            # 人脸大小得分
            100                                          # 转换为百分比
        )
        
        print(f"    人脸检测详情:")
        print(f"      检测置信度: {best_detection.score[0]*100:.2f}% (权重: {detection_weight})")
        print(f"      人脸框宽度: {bbox.width:.4f}")
        print(f"      人脸框高度: {bbox.height:.4f}")
        print(f"      宽高比(w/h): {aspect_ratio:.2f} (理想: {ideal_ratio})")
        print(f"      高宽比(h/w): {(bbox.height/bbox.width):.2f}")
        print(f"      宽高比得分: {ratio_score*100:.2f}% (权重: {ratio_weight})")
        print(f"      人脸大小得分: {face_size_score*100:.2f}% (权重: {size_weight})")
        print(f"      人脸区域占比: {face_area*100:.2f}%")
        
        return confidence

    def process(self, video, confidence_threshold, frame_skip, test_first_frame=False):
        if test_first_frame:
            return self.process_first_frame(video, confidence_threshold, frame_skip)
        
        try:
            # 添加视频预览
            self.preview_video(video)
            
            # 构建完整的视频路径
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.isfile(video_path):
                print(f"警告: 找不到视频文件: {video}")
                return (np.zeros((512, 512, 3), dtype=np.uint8),)  # 返回黑色图像

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频文件: {video}")
                return (np.zeros((512, 512, 3), dtype=np.uint8),)  # 返回黑色图像

            best_frame = None
            best_confidence = 0
            frame_count = 0
            
            print(f"开始处理视频: {video}")
            print(f"置信度阈值: {confidence_threshold}")
            print(f"帧跳过间隔: {frame_skip}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                print(f"\n处理第 {frame_count} 帧:")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测姿态和人脸
                pose_results = self.pose.process(frame_rgb)
                face_results = self.face_detection.process(frame_rgb)
                
                # 计算置信度
                pose_confidence = self.is_frontal_pose(pose_results)
                face_confidence = self.is_frontal_face(face_results)
                
                # 打印详细的置信度信息
                print(f"  姿态置信度: {pose_confidence:.2f}%")
                print(f"  人脸置信度: {face_confidence:.2f}%")
                
                # 综合置信度
                total_confidence = min(pose_confidence, face_confidence)
                print(f"  综合置信度: {total_confidence:.2f}%")
                
                # 更新最佳帧
                if total_confidence > best_confidence:
                    best_confidence = total_confidence
                    best_frame = frame_rgb
                    print(f"  >>> 更新最佳帧，当前最佳置信度: {best_confidence:.2f}%")
                
                # 如果达到阈值则提前结束
                if best_confidence >= confidence_threshold:
                    print(f"\n已达到置信度阈值，提前结束处理")
                    break
            
            cap.release()
            
            if best_frame is None:
                print(f"警告: 未能找到符合条件的帧")
                # 创建空图像并转换为 PyTorch 张量
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.release()
                
                if ret:
                    h, w = first_frame.shape[:2]
                    empty_frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    empty_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                
                # 转换为 PyTorch 张量
                empty_tensor = torch.from_numpy(empty_frame).float() / 255.0
                return (empty_tensor,)
            
            print(f"\n处理完成:")
            print(f"总共处理帧数: {frame_count}")
            print(f"最终最佳置信度: {best_confidence:.2f}%")
            
            # 将最佳帧转换为 PyTorch 张量
            best_frame_tensor = torch.from_numpy(best_frame).float() / 255.0
            return (best_frame_tensor,)
            
        except Exception as e:
            print(f"\n处理出错: {str(e)}")
            # 返回黑色图像并转换为 PyTorch 张量
            empty_tensor = torch.zeros((512, 512, 3)).float()
            return (empty_tensor,)
        
    @classmethod
    def VALIDATE_INPUTS(cls, video, confidence_threshold, frame_skip):
        # 验证视频文件
        if not folder_paths.exists_annotated_filepath(video):
            return f"无效的视频文件: {video}"
        
        # 验证置信度阈值
        if not (0 <= confidence_threshold <= 100):
            return f"置信度阈值必须在 0-100 之间，当前值: {confidence_threshold}"
        
        # 验证帧跳过参数
        if frame_skip < 1:
            return f"帧跳过参数必须大于 0，当前值: {frame_skip}"
        
        return True
    
    @classmethod
    def IS_CHANGED(cls, video, confidence_threshold, frame_skip):
        # 获取视频文件的完整路径
        video_path = folder_paths.get_annotated_filepath(video)
        if not video_path:
            return ""
        
        # 将所有参数都纳入考虑
        m = hashlib.sha256()
        m.update(str(os.path.getmtime(video_path)).encode())  # 文件修改时间
        m.update(str(confidence_threshold).encode())           # 置信度阈值
        m.update(str(frame_skip).encode())                    # 帧跳过参数
        
        return m.hexdigest()
    
    # 添加视频预览方法
    def preview_video(self, video):
        video_path = os.path.join(folder_paths.get_input_directory(), video)
        if os.path.exists(video_path):
            return {
                "widget": "videoplayer",
                "src": f"file={video_path}",
                "width": 400,  # 可以调整视频播放器的宽度
                "height": 300  # 可以调整视频播放器的高度
            }
        return None
    
    def process_first_frame(self, video, confidence_threshold, frame_skip):
        """仅处理第一帧的测试方法"""
        try:
            # 构建完整的视频路径
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.isfile(video_path):
                print(f"警告: 找不到视频文件: {video}")
                return (torch.zeros((512, 512, 3)).float(),)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频文件: {video}")
                return (torch.zeros((512, 512, 3)).float(),)

            print(f"\n开始处理视频第一帧: {video}")
            
            # 只读取第一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("无法读取视频帧")
                return (torch.zeros((512, 512, 3)).float(),)
            
            # 打印原始帧的信息
            print(f"原始帧大小: {frame.shape}")
            print(f"帧数据类型: {frame.dtype}")
            print(f"帧数值范围: [{frame.min()}, {frame.max()}]")
            
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为 PyTorch 张量
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            print(f"转换后张量大小: {frame_tensor.shape}")
            print(f"张量数值范围: [{frame_tensor.min().item():.3f}, {frame_tensor.max().item():.3f}]")
            
            return (frame_tensor,)
                
        except Exception as e:
            print(f"\n处理出错: {str(e)}")
            return (torch.zeros((512, 512, 3)).float(),)
    
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


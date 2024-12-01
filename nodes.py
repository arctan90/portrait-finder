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
                    "default": 70.0,
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
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        
        # 计算垂直对齐度（左右方向）
        horizontal_alignment = abs(nose.x - mid_hip_x)
        
        # 检查躯干是否垂直（新增）
        # 计算躯干与垂直线的夹角
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        trunk_vertical_diff = abs((mid_shoulder_y - mid_hip_y) - 
                                abs(left_shoulder.y - left_hip.y))  # 理想情况下应接近0
        
        # 计算各项得分
        shoulder_score = (1.0 - shoulder_diff) * 100  # 肩膀平行度得分
        horizontal_score = (1.0 - horizontal_alignment) * 100  # 水平对齐得分
        vertical_score = (1.0 - trunk_vertical_diff) * 100  # 垂直站姿得分
        
        # 计算总体置信度
        confidence = min(shoulder_score, horizontal_score, vertical_score)
        
        # 打印详细信息
        print(f"    姿态检测详情:")
        print(f"      左肩位置: ({left_shoulder.x:.3f}, {left_shoulder.y:.3f}, {left_shoulder.z:.3f})")
        print(f"      右肩位置: ({right_shoulder.x:.3f}, {right_shoulder.y:.3f}, {right_shoulder.z:.3f})")
        print(f"      肩膀深度差异: {shoulder_diff:.3f}")
        print(f"      肩膀对齐得分: {shoulder_score:.2f}%")
        print(f"      鼻子位置: ({nose.x:.3f}, {nose.y:.3f})")
        print(f"      躯干中点: ({mid_hip_x:.3f}, {mid_hip_y:.3f})")
        print(f"      水平对齐偏差: {horizontal_alignment:.3f}")
        print(f"      水平对齐得分: {horizontal_score:.2f}%")
        print(f"      躯干垂直偏差: {trunk_vertical_diff:.3f}")
        print(f"      垂直站姿得分: {vertical_score:.2f}%")
        print(f"      最终姿态得分: {confidence:.2f}%")
        
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
        print(f"      宽高比得分: {ratio_score*100:.2f}% (权重: {ratio_weight})")
        print(f"      人脸大小得分: {face_size_score*100:.2f}% (权重: {size_weight})")
        print(f"      人脸区域占比: {face_area*100:.2f}%")
        
        return confidence

    def get_empty_frame(self, video_path):
        """获取与视频相同分辨率的空图像"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return torch.zeros((512, 512, 3)).float()  # 默认尺寸作为后备
            
            # 读取第一帧以获取尺寸
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return torch.zeros((512, 512, 3)).float()  # 默认尺寸作为后备
            
            # 创建与原视频相同分辨率的空图像
            # 注意：需要考虑旋转后的尺寸
            h, w = frame.shape[:2]
            empty_frame = np.zeros((w, h, 3), dtype=np.uint8)  # 交换宽高以匹配旋转后的尺寸
            return torch.from_numpy(empty_frame).float()
            
        except Exception as e:
            print(f"创建空图像时出错: {str(e)}")
            return torch.zeros((512, 512, 3)).float()  # 默认尺寸作为后备

    def process(self, video, confidence_threshold, frame_skip, test_first_frame=False):
        if test_first_frame:
            return self.process_first_frame(video, confidence_threshold, frame_skip)
        
        try:
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.isfile(video_path):
                print(f"警告: 找不到视频文件: {video}")
                return (self.get_empty_frame(video_path),)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频文件: {video}")
                return (self.get_empty_frame(video_path),)

            frame_count = 0
            target_frame_index = -1  # 用于存储符合条件的帧的索引
            print(f"开始处理视频: {video}")
            print(f"置信度阈值: {confidence_threshold}")
            
            # 第一次遍历：寻找符合条件的帧的索引
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                # 旋转和处理图像
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测姿态和人脸
                pose_results = self.pose.process(frame_rgb)
                face_results = self.face_detection.process(frame_rgb)
                
                pose_confidence = self.is_frontal_pose(pose_results)
                face_confidence = self.is_frontal_face(face_results)
                total_confidence = min(pose_confidence, face_confidence)
                
                print(f"\n第 {frame_count} 帧 - 综合置信度: {total_confidence:.2f}%")
                
                # 如果找到符合条件的帧，记录索引并停止
                if total_confidence >= confidence_threshold:
                    print(f"找到符合条件的帧，置信度: {total_confidence:.2f}%")
                    target_frame_index = frame_count
                    break
            
            cap.release()
            
            # 如果没找到符合条件的帧，返回空图像
            if target_frame_index == -1:
                print("未找到符合条件的帧，返回空图像")
                return (self.get_empty_frame(video_path),)
            
            # 第二次打开视频：直接读取目标帧
            print(f"正在提取第 {target_frame_index} 帧...")
            cap = cv2.VideoCapture(video_path)
            current_frame = 0
            found_frame = None
            
            while current_frame < target_frame_index:
                ret = cap.read()[0]  # 只需要 ret 值
                if not ret:
                    break
                current_frame += 1
            
            # 读取目标帧
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # 旋转图像
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
                # 转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为张量
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                print(f"成功提取目标帧，形状: {frame_tensor.shape}")
                return (frame_tensor,)
            else:
                print("提取目标帧失败")
                return (self.get_empty_frame(video_path),)
                
        except Exception as e:
            print(f"处理出错: {str(e)}")
            return (self.get_empty_frame(video_path),)
    
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
                "height": 300  # 可以调整视播放器的高度
            }
        return None
    
    def process_first_frame(self, video, confidence_threshold, frame_skip):
        """仅处理第一帧的测试方法"""
        try:
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.isfile(video_path):
                print(f"警告: 找不到视频文件: {video}")
                return (self.get_empty_frame(video_path),)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频文件: {video}")
                return (self.get_empty_frame(video_path),)

            print(f"\n读取视频第一帧: {video}")
            
            # 读取第一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("无法读取视频帧")
                return (self.get_empty_frame(video_path),)
            
            # 打印原始帧的信息
            print(f"原始帧大小: {frame.shape}")
            
            # 旋转图像以匹配原始视频方向
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
            print(f"旋转后帧大小: {frame.shape}")
            
            # 转换颜色空间并返回
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            return (frame_tensor,)
                
        except Exception as e:
            print(f"\n处理出错: {str(e)}")
            return (self.get_empty_frame(video_path),)
    
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


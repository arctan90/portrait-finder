import os
import sys
import subprocess
import importlib.util

# 检查必需的库
REQUIRED_PACKAGES = [
    ("mediapipe", "mediapipe>=0.10.0"),
    ("cv2", "opencv-python>=4.8.0"),
    ("numpy", "numpy>=1.24.0"),
]

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 检查并安装缺失的包
for package_name, package_spec in REQUIRED_PACKAGES:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"Installing {package_spec}...")
        try:
            install_package(package_spec)
        except Exception as e:
            print(f"Error installing {package_spec}: {str(e)}")

# 导入节点类
from .nodes import VideoFrontalDetectorNode

# 获取版本号
with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as f:
    __version__ = f.read().strip()

# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoFrontalDetector": VideoFrontalDetectorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrontalDetector": "人物正面检测"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
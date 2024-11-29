from .nodes import VideoFrontalDetectorNode

NODE_CLASS_MAPPINGS = {
    "VideoFrontalDetectorNode": VideoFrontalDetectorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrontalDetectorNode": "人物正面检测"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
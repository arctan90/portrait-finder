# ComfyUI Portrait Finder

一个 ComfyUI 插件，可以从视频中智能检测并提取最佳人物正面肖像帧。

## 功能特点

- 智能识别最佳人物正面肖像
- 自动检测人物正面姿态
- 可调节检测置信度阈值
- 支持常见视频格式

## 安装步骤

1. 确保已安装 ComfyUI

2. 进入 ComfyUI 的 custom_nodes 目录：
```bash
cd ComfyUI/custom_nodes
```

3. 克隆本插件：
```bash
git clone https://github.com/arctan90/portrait-finder.git
```

4. 安装依赖：
```bash
pip install -r portrait-finder/requirements.txt
```

5. 重启 ComfyUI

## 部署验证

1. 启动 ComfyUI 后，检查节点列表中是否出现"人物正面检测"节点
2. 如果没有出现，请检查：
   - custom_nodes 目录下是否有 portrait-finder 文件夹
   - 依赖是否安装成功（可以重新运行 pip install 命令）
   - ComfyUI 控制台是否有报错信息

## 使用方法

1. 启动 ComfyUI
2. 在节点列表中找到 "人物正面检测"
3. 设置参数：
   - video: 上传视频文件
   - video_file: 选择 input 目录中的视频文件
   - use_uploaded_video: 选择使用上传的视频还是已有视频文件
   - confidence_threshold: 置信度阈值（0-100，默认80）

## 视频输入方式

支持两种视频输入方式：
1. 直接上传视频文件
2. 从 ComfyUI 的 input 目录选择视频文件（支持 mp4, avi, mov, mkv 格式）

## 参数说明

| 参数名 | 类型 | 说明 | 默认值 | 范围 |
|--------|------|------|--------|------|
| video | VIDEO | 上传视频文件 | - | - |
| video_file | VIDEO | 选择 input 目录中的视频文件 | - | - |
| use_uploaded_video | BOOLEAN | 选择使用上传的视频还是已有视频文件 | True | - |
| confidence_threshold | FLOAT | 检测置信度阈值 | 80.0 | 0-100 |

## 输出

- IMAGE: 检测到的最佳正面帧

## 注意事项

1. 确保视频文件路径正确且可访问
2. 视频文件支持常见格式（mp4, avi, mov等）
3. 处理大文件时可能需要一定时间

## 故障排除

1. 如果提示 "无法打开视频文件"：
   - 检查视频路径是否正确
   - 确认视频格式是否支持
   - 验证文件访问权限

2. 如果提示 "未能找到符合条件的帧"：
   - 尝试降低 confidence_threshold 值
   - 确认视频中是否包含正面人物画面

3. 如果遇到依赖相关错误：
```bash
pip install --upgrade opencv-python mediapipe numpy
```

## 更新说明

如需更新插件：
1. 进入插件目录：
```bash
cd ComfyUI/custom_nodes/portrait-finder
```

2. 拉取最新代码：
```bash
git pull
```

3. 更新依赖：
```bash
pip install -r requirements.txt --upgrade
```

4. 重启 ComfyUI

## 技术支持

如果遇到问题，请：
1. 检查 ComfyUI 控制台输出
2. 查看上述故障排除步骤
3. 提交 Issue 到项目仓库

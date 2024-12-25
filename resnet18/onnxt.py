import onnx
import onnxruntime as ort
import numpy as np
import cv2
from torchvision import transforms

# 数据增强或标准化
transform = transforms.Compose([
    transforms.ToTensor()  # 转换为Tensor
])

# 加载ONNX模型
onnx_model = onnx.load("lane_center_model.onnx")
onnx.checker.check_model(onnx_model)  # 检查模型是否有效

# 设置ONNX运行时
ort_session = ort.InferenceSession("lane_center_model.onnx")

# 测试输入数据 (假设输入尺寸是 224x224x3)
# 加载一张测试图像
image = cv2.imread('./data/test/image12.jpg')

image = transform(image)

# 确保图像是 float32 类型
image = image.float()

# 增加 batch 维度 -> (1, C, H, W)
image = np.expand_dims(image, axis=0)

# 创建输入字典
inputs = {ort_session.get_inputs()[0].name: image}

# 进行预测
outputs = ort_session.run(None, inputs)

# 获取模型输出
print("Model output:", outputs)

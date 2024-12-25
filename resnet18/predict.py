import os
import torch
import torchvision.transforms as transforms
import PIL.Image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import uuid
import logging
import time

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("prediction_log.txt"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger()

# 模型加载函数
def load_model(model_path, device):
    """加载保存的最佳模型"""
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 图片处理函数
def preprocess_image(image_path):
    """将图片预处理为模型输入格式"""
    image = PIL.Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加批量维度
    return image

# 从文件名中提取坐标
def extract_coordinates_from_filename(filename):
    """从图片文件名中提取x和y坐标"""
    name_parts = filename.split('_')
    x_value = float(name_parts[1])  # 假设第二个部分是x坐标
    y_value = float(name_parts[2])  # 假设第三个部分是y坐标
    return x_value, y_value

# 主测试函数
def test_images_in_folder(image_folder, model_path, device, output_folder):
    """对文件夹中的所有图片进行测试，并打印出预测的x坐标和标记的x坐标"""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图片文件夹不存在：{image_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载模型
    model = load_model(model_path, device)

    # 初始化误差累加器和时间统计器
    total_error = 0
    num_images = 0
    total_time = 0

    # 遍历图片文件夹
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):  # 确保是图片文件
            image_path = os.path.join(image_folder, image_name)

            # 从文件名中提取标记的坐标
            label_x, label_y = extract_coordinates_from_filename(image_name)

            # 预处理图片
            image = preprocess_image(image_path)
            image = image.to(device)

            # 预测
            start_time = time.time()
            with torch.no_grad():
                output = model(image)
                predicted_coords = output.cpu().numpy()[0]
            end_time = time.time()

            # 计算时间
            processing_time = end_time - start_time
            total_time += processing_time

            # 计算预测误差（x坐标）
            x_pred = (predicted_coords[0] * 176 / 2) + (176 / 2)
            error = abs(x_pred - label_x)  # 计算绝对误差

            # 累加误差
            total_error += error
            num_images += 1

            # 打印预测的坐标和标记的坐标
            logger.info(f"image:{image_name}")
            logger.info(f"predict_x:{x_pred}")
            logger.info(f"label_x:{label_x}")
            logger.info(f"error:{error}")
            logger.info(f"processing_time:{processing_time:.6f} seconds")

            # 保存预测结果
            unique_id = f"xy_{predicted_coords[0]:.2f}_{predicted_coords[1]:.2f}_{uuid.uuid1()}"
            output_path = os.path.join(output_folder, f"{unique_id}.jpg")
            PIL.Image.open(image_path).save(output_path)

    # 计算并打印平均误差和平均时间
    logger.info("Total number of images in the test set: %d", num_images)
    if num_images > 0:
        avg_error = total_error / num_images
        acc_error = avg_error / 176 * 100
        avg_time = total_time / num_images
        logger.info(f"The average error (x-coordinate) of all images: {avg_error:.2f} pixel")
        logger.info(f"The model accuracy: {acc_error:.2f}%")
        logger.info(f"The average processing time per image: {avg_time:.6f} seconds")
    else:
        logger.warning("No pictures were processed")

if __name__ == "__main__":
    # 配置文件夹路径和模型路径
    image_folder = './test_label'  # 替换为你的图片文件夹路径
    best_model_path = './model_v1/best_model_v1.pth'  # 替换为你的最佳模型路径
    output_folder = './predicted_results'  # 存放预测结果的文件夹
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_images_in_folder(image_folder, best_model_path, device, output_folder)

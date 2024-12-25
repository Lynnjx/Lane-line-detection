import os
import re
import matplotlib.pyplot as plt

# 文件夹路径
input_folder = './image_label'

# 检查输入文件夹是否存在
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"输入文件夹 {input_folder} 不存在！")

# 提取 x 值的正则表达式
x_pattern = re.compile(r"xy_(\d{3})_\d{3}_")

def extract_x_values(folder):
    """从文件名中提取 x 值"""
    x_values = []
    for filename in os.listdir(folder):
        match = x_pattern.search(filename)
        if match:
            x_values.append(int(match.group(1)))
    return x_values

def plot_histogram(x_values):
    """绘制直方图"""
    if not x_values:
        print("未找到有效的 x 值，无法绘制直方图！")
        return
    plt.figure(figsize=(12, 6))
    
    plt.hist(x_values, bins=20, color='blue', edgecolor='black')

    # 设置标题和轴标签，并调整字体大小
    plt.title('X-axis Annotation Distribution', fontsize=18)
    plt.xlabel('X Coordinate', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)

    # 调整刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()

if __name__ == "__main__":
    x_values = extract_x_values(input_folder)
    plot_histogram(x_values)

import os
import cv2 as cv
import uuid
import numpy as np
import matplotlib.pyplot as plt

# 文件夹路径
input_folder = './test'
output_folder = './label_test'

# 检查输入文件夹是否存在
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"输入文件夹 {input_folder} 不存在！")

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 全局变量
x, y = -1, -1  # 用于记录标注位置
current_image = None
image_with_circle = None
x_values = []  # 用于记录标注的x轴数据

def update_histogram():
    """更新直方图"""
    plt.clf()
    plt.hist(x_values, bins=20, color='blue', edgecolor='black')
    plt.title('X-axis Annotation Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Frequency')
    plt.pause(0.1)

def mouse_callback(event, x_coord, y_coord, flags, userdata):
    """鼠标事件回调函数，用于标注位置"""
    global x, y, image_with_circle
    if event == cv.EVENT_LBUTTONDOWN:
        x, y = x_coord, y_coord
        image_with_circle = current_image.copy()
        cv.circle(image_with_circle, (x, y), 5, (0, 0, 255), -1)
        cv.putText(image_with_circle, f'({x}, {y})', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Image Annotation", image_with_circle)

def annotate_images():
    """主函数：读取图片，标注位置并保存"""
    global x, y, current_image, image_with_circle

    # 获取输入文件夹中的所有图片
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print("输入文件夹中没有找到图片文件！")
        return

    plt.ion()  # 开启实时绘图
    plt.figure(figsize=(8, 6))

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        current_image = cv.imread(image_path)

        if current_image is None:
            print(f"无法读取图片：{image_file}")
            continue

        # 显示图片
        cv.imshow("Image Annotation", current_image)
        cv.setMouseCallback("Image Annotation", mouse_callback)

        # 等待用户标注
        while True:
            key = cv.waitKey(1)
            if key == 13:  # 按下回车键保存标注
                if x != -1 and y != -1:
                    unique_id = f"xy_{x:03d}_{y:03d}_{uuid.uuid1()}"
                    output_path = os.path.join(output_folder, f"{unique_id}.jpg")
                    # 保存标注后的图片
                    if image_with_circle is not None:
                        cv.imwrite(output_path, image_with_circle)
                    else:
                        cv.imwrite(output_path, current_image)
                    print(f"标注已保存至：{output_path}")

                    # 更新直方图
                    x_values.append(x)
                    update_histogram()

                    # 重置标注坐标
                    x, y = -1, -1
                    break
            elif key == 27:  # 按下ESC键跳过当前图片
                print(f"跳过图片：{image_file}")
                break

    # 销毁窗口
    plt.ioff()  # 关闭实时绘图
    plt.show()
    cv.destroyAllWindows()

if __name__ == "__main__":
    annotate_images()

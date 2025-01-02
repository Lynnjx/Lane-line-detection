# Intelligent-car-system-based-on-Horizon-and-ROS2-implementation

Demo Video:https://www.bilibili.com/video/BV1jA6zY1EiE/?vd_source=54b64d3cd509c0c5b984cdd2008ef9a6

## Introduction

The development board used in this project is the Sunrise X3 provided by Horizon.

The centering function of the trolley based on the ROS2 framework and implemented using the Resnet18 model.

Using the OE development environment provided by Horizon, the model is quantized and converted to bin format, and the final quantized model achieved reduces the processing time to 0.0090s for each image frame.

|Test Parameters \ Format(Harware Platform)|pth(Intel 13)|onnx(X3)|bin(X3)|
|---|---|---|---|
|Mean Pixel Error (%)|1.72%|1.72%|8.04%|
|Average Inference Time (s)|0.0203|0.2863|**0.0090**|

## Documentation

1. `10_auto_run`: This document is based on the OE tool chain and is written as a framework for model testing, calibration, and quantification
2. `auto_run`: Under this file, all functional packages under the ros2 framework are provided.
3. `resnet18`: Under this file, we provide all the information of the model training and testing.

## More details

1. Go to the official website and download the OE toolchain and the corresponding docker environment, make sure to put the file `10_auto_run` in the folder 03_classification.
2. The version of ROS2 used in this project is humble, match the corresponding version and then execute the following command to start the project code.
```shell
colcon build # Compile all projects
source install/setup.bash # Add dependencies
ros2 launch patrol_line patrol_line.launch.py # Start all nodes
```
3. In ros, the call paths of the model, the UART communication with the host computer, and the camera interface need to be modified on their own terms.

## References

1. Open Robotics, ROS 2 Documentation: Humble, https://docs.ros.org/en/humble/index.html.
2. D_robotics 开 发 者 社 区, 深 度 学 习 巡 线 小 车, https://developer.d-robotics.cc/rdk_doc/Application_case/line_follower.
3. D_robotics 开 发 者 社 区, 玩 转 地 平 线 工 具 链 竟 如 此 简 单！ , https://www.bilibili.com/video/BV1Xh411P73Z/?spm_id_from=333.337.search-card.all.click&vd_source=54b64d3cd509c0c5b984cdd2008ef9a6.
4. D_robotics 开 发 者 社 区, OpenExplorer 算 法 工 具 链 开 发, https://developer.horizon.ai:8005/api/v1/fileData/documents_pi/ai_toolchain_develop/preface_toolchain_overview.html

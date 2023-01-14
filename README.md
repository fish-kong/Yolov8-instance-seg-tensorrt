# Yolov8-instance-seg-tensorrt
based on the yolov8，provide pt-onnx-tensorrt transcode and infer code by c++

mkdir build  
cd build  
cmake ..  
make  
sudo ./onnx2trt ../models/yolov8n-seg.onnx ../models/yolov8n-seg.engine  
sudo ./trt_infer ../models/yolov8n-seg.onnx ../images/bus.jpg  
![image](https://github.com/fish-kong/Yolov8-instance-seg-tensorrt/blob/main/x.jpg)

# rscnn
This is a CNN framework based on RenderScript.

### Usage
1. Download [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) model.
```
git clone https://github.com/chuanqi305/MobileNet-SSD
```
2. Use script/convert_caffe_model.py to convert the model to new format, do not forget to change the caffe root path in the converting script.
```
python script/convert_caffe_model.py --model MobileNet-SSD/deploy.prototxt --weights MobileNet-SSD/MobileNetSSD_deploy.caffemodel --savedir mobilenet-ssd
```
3. Push the converted model files to your mobile phone.
```
adb push mobilenet-ssd /sdcard/
```
Instead of push the files to mobile phone manually, you can also put these files into the assets of the project:
```
cp -ar mobilenet-ssd demo/src/main/assets/
```
4. Run this demo, and you can select a photo to see the object detection result.

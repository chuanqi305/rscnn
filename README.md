# rscnn
This is a CNN framework based on RenderScript.

### Usage
1. Download (MobileNet-SSD)[https://github.com/chuanqi305/MobileNet-SSD] model.
```
git clone https://github.com/chuanqi305/MobileNet-SSD
```
2. Convert the model to new format by script/convert_caffe_model.py ,do not forget to change the caffe root path in the script.
```
python script/convert_caffe_model.py --model MobileNet-SSD/deploy.prototxt --weights MobileNet-SSD/MobileNetSSD_deploy.caffemodel --savedir mobilenet-ssd
```
3. Push the converted model files to your mobile phone.
```
adb push mobilenet-ssd /sdcard/
```
Instead of push the files to mobile phone manually, you can also put the converted model files into the assets of the demo:
```
cp -ar mobilenet-ssd demo/src/main/assets/
```
4. Run this demo, and you can select a photo to see the object detection result.

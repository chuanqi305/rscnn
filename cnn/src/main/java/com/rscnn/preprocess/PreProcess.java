package com.rscnn.preprocess;

import android.graphics.Bitmap;

import com.rscnn.postprocess.NetworkParameter;

public class PreProcess {
    protected ROI roi = null;
    protected float[] meanValueBGR = new float[]{102.9801f, 115.9465f, 122.7717f};
    protected float scale = 1.f;

    public PreProcess() {
    }

    public PreProcess(float[] meanValueBGR, float scale) {
        this.meanValueBGR = meanValueBGR;
        this.scale = scale;
    }

    public PreProcess(ROI roi, float[] meanValueBGR, float scale) {
        this.roi = roi;
        this.meanValueBGR = meanValueBGR;
        this.scale = scale;
    }

    public Object[] process(Bitmap image, NetworkParameter param) {
        float pixelMeanBlue = meanValueBGR[0];
        float pixelMeanGreen = meanValueBGR[1];
        float pixelMeanRed = meanValueBGR[2];

        int height = param.getNetworkInputHeight();
        int width = param.getNetworkInputWidth();

        float[] data = new float[height * width * 4];
        int[] pixels = new int[height * width];

        Bitmap subImage;
        if (roi !=null && (roi.getX() != 0 || roi.getY() !=0 || roi.getHeight() != image.getHeight()
                || roi.getWidth() != image.getWidth())) {
            subImage = Bitmap.createBitmap(image, roi.getX(), roi.getY(),
                    roi.getWidth(), roi.getHeight());
        }
        else {
            subImage = image;
        }

        if(subImage.getWidth()!= width || subImage.getHeight() != height){
            Bitmap bmp2 = Bitmap.createScaledBitmap(subImage, width, height, false);
            bmp2.getPixels(pixels, 0, width, 0, 0, width, height);
            bmp2.recycle();
        }
        else{
            subImage.getPixels(pixels, 0, width, 0, 0, width, height);
        }

        int count = 0;
        int dataCount = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int k = pixels[count++];
                data[dataCount++] = ((k & 0xFF) - pixelMeanBlue) * scale;
                data[dataCount++] = (((k >> 8) & 0xFF) - pixelMeanGreen) * scale;
                data[dataCount++] = (((k >> 16) & 0xFF) - pixelMeanRed) * scale;
                data[dataCount++] = 0;
            }
        }

        if (subImage != image) {
            subImage.recycle();
        }

        return new Object[]{data};
    }
}

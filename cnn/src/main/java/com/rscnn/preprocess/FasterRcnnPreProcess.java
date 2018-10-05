package com.rscnn.preprocess;

import android.graphics.Bitmap;

import com.rscnn.postprocess.NetworkParameter;
import com.rscnn.utils.LogUtil;

public class FasterRcnnPreProcess extends PreProcess {

    private float RPN_NMS_THRESH = 0.4f;
    private int RPN_PRE_NMS_TOP_N = 6000;
    private int RPN_POST_NMS_TOP_N = 200;

    private String TAG = "FasterRcnnPreProcess";

    public FasterRcnnPreProcess(float[] meanValueBGR, float scale) {
        super(meanValueBGR, scale);
    }

    @Override
    public Object[] process(Bitmap image, NetworkParameter param){

        int origHeight = image.getHeight();
        int origWidth = image.getWidth();
        int targetHeight = param.getNetworkInputHeight();
        int targetWidth = param.getNetworkInputWidth();

        Object[] data = super.process(image, param);

        float scaleX = (float)targetWidth / (float) origHeight;
        float scaleY = (float)targetHeight / (float) origWidth;
        float[] iminfo = new float[]{targetHeight,targetWidth, scaleX, scaleY,
                RPN_PRE_NMS_TOP_N,RPN_POST_NMS_TOP_N,RPN_NMS_THRESH
        };
        return new Object[]{data[0], iminfo};
    }
}

package com.rscnn.model;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.renderscript.RenderScript;

import com.rscnn.network.ConvNet;
import com.rscnn.network.DetectResult;
import com.rscnn.postprocess.FasterRcnnPostProcess;
import com.rscnn.postprocess.PostProcess;
import com.rscnn.postprocess.SSDPostProcess;
import com.rscnn.preprocess.FasterRcnnPreProcess;
import com.rscnn.preprocess.PreProcess;
import com.rscnn.utils.LogUtil;

import java.io.IOException;
import java.util.List;

public class PvaLite extends ObjectDetector {

    private PreProcess preProcess;
    private PostProcess postProcess;
    private int SCALE = 640;
    private float SCALE_MULTIPLE_OF = 32.f;
    private int MAX_SIZE = 1000;

    public PvaLite(RenderScript renderScript, AssetManager assetManager, String modelDir) throws IOException {
        float[] meanValue = new float[]{102.9801f, 115.9465f, 122.7717f};
        preProcess = new FasterRcnnPreProcess(meanValue, 1.0f);
        postProcess = new FasterRcnnPostProcess();
        this.convNet = new ConvNet(renderScript, assetManager, modelDir, preProcess, postProcess);
    }

    private int[] computeNetworkInputSize(int bmpWidth, int bmpHeight){
        float targetSize = SCALE;
        float maxSize = MAX_SIZE;
        float scaleMultiple = SCALE_MULTIPLE_OF;
        float imSizeMin = Math.min(bmpWidth, bmpHeight);
        float imSizeMax = Math.max(bmpWidth, bmpHeight);
        float imScale = targetSize / imSizeMin;

        if((imSizeMax * imScale) > maxSize){
            imScale = maxSize / imSizeMax;
        }

        float scaleX = (float)Math.floor(bmpWidth * imScale /scaleMultiple) * scaleMultiple / (float)bmpWidth;
        float scaleY = (float)Math.floor(bmpHeight * imScale /scaleMultiple) * scaleMultiple / (float)bmpHeight;
        int height = (int)((float)bmpHeight * scaleY);
        int width =(int)( (float)bmpWidth * scaleX);
        return new int[]{width, height};
    }

    @Override
    public List<DetectResult> detect(Bitmap image) {
        int networkInputHeight = convNet.getNetworkInputHeight();
        int networkInputWidth = convNet.getNetworkInputWidth();
        int[] needSize = computeNetworkInputSize(image.getWidth(), image.getHeight());
        if (needSize[0] != networkInputWidth || needSize[1] != networkInputHeight) {
            LogUtil.i("PvaLite", "resizing network input shape to " + needSize + " * " + needSize[1]);
            convNet.resetNetworkInputSize(needSize[0], needSize[1]);
        }
        return super.detect(image);
    }
}

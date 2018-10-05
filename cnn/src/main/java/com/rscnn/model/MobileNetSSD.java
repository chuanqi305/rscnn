package com.rscnn.model;

import android.content.res.AssetManager;
import android.renderscript.RenderScript;

import com.rscnn.network.ConvNet;
import com.rscnn.network.Layer;
import com.rscnn.postprocess.PostProcess;
import com.rscnn.postprocess.SSDPostProcess;
import com.rscnn.preprocess.PreProcess;

import java.io.IOException;

public class MobileNetSSD extends ObjectDetector {

    static {
        Layer.setReshapeFromTensorFlow(true);
    }

    public MobileNetSSD(RenderScript renderScript, AssetManager assetManager, String modelDir) throws IOException {
        float[] meanValue = new float[]{127.5f,127.5f,127.5f};
        PreProcess preProcess = new PreProcess(meanValue, 0.007843f);
        PostProcess postProcess = new SSDPostProcess();
        this.convNet = new ConvNet(renderScript, assetManager, modelDir, preProcess, postProcess);
    }
}

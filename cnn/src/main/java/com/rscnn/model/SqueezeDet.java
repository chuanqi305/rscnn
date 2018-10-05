package com.rscnn.model;

import android.content.res.AssetManager;
import android.renderscript.RenderScript;

import com.rscnn.network.ConvNet;
import com.rscnn.network.Layer;
import com.rscnn.postprocess.PostProcess;
import com.rscnn.postprocess.SqueezeDetPostProcess;
import com.rscnn.preprocess.PreProcess;

import java.io.IOException;

public class SqueezeDet extends ObjectDetector {

    static {
        Layer.setWeightFromTensorFlow(true);
    }

    public SqueezeDet(RenderScript renderScript, AssetManager assetManager, String modelDir) throws IOException {
        float[] meanValue = new float[]{103.939f,116.779f,123.68f};
        PreProcess preProcess = new PreProcess(meanValue, 1.0f);
        PostProcess postProcess = new SqueezeDetPostProcess();
        this.convNet = new ConvNet(renderScript, assetManager, modelDir, preProcess, postProcess);
    }
}

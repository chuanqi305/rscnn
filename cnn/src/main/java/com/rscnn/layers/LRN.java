package com.rscnn.layers;

import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_LRN;

public class LRN extends Layer {

    private int localSize = 5;                      // local size
    private float alpha = 1.f;                       // alpha
    private float beta = 0.75f;                        // beta
    private String normRegion = "ACROSS_CHANNELS";                  // norm region: "across_channels"
    private float k = 1.f;

    private static final String TAG = "LRN";
    private ScriptC_LRN lrnScript;

    public void setLocalSize(int localSize) {
        this.localSize = localSize;
    }

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    public void setBeta(float beta) {
        this.beta = beta;
    }

    public void setK(float k) {
        this.k = k;
    }

    public void setNormRegion(String normRegion) {
        this.normRegion = normRegion.toUpperCase();
    }

    public LRN() {
    }

    @Override
    public void setup(){
        lrnScript = new ScriptC_LRN(renderScript);
        float alphaDivideN = alpha / ((float) localSize * (float) localSize);
        lrnScript.set_alpha_divide_n(alphaDivideN);
        lrnScript.set_beta(beta);
        lrnScript.set_local_size(localSize);
        lrnScript.set_k(k);
        int height = inputShape[0][1];
        int width = inputShape[0][2];
        int channel = inputShape[0][3];
        int channelAligned = channel;
        if(channel % 4 !=0 ){
            channelAligned = channel + 4 - channel % 4;
        }
        lrnScript.set_height(height);
        lrnScript.set_width(width);
        lrnScript.set_channel(channel);
        lrnScript.set_channelAligned(channelAligned);
    }

    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
        allocFeatureMap();
    }

    @Override
    public void computeFeatureMap(){
        FeatureMap input = (FeatureMap) featureMapInput[0];
        FeatureMap output = (FeatureMap) featureMapOutput;
        lrnScript.set_in_blob(input.getFeatureMap());
        lrnScript.set_out_blob(output.getFeatureMap());
        Script.LaunchOptions option = new Script.LaunchOptions();
        int channel = outputShape[3];
        int channelAligned = getInputChannelAligned();
        switch (normRegion) {
            case "ACROSS_CHANNELS":
                option.setX(0, channel).setY(0, outputShape[1] * outputShape[2]);
                lrnScript.forEach_cross_channel(option);
                break;
            case "WITHIN_CHANNEL":
                option.setX(0, channelAligned / 4).setY(0, outputShape[1] * outputShape[2]);
                lrnScript.forEach_within_channel(option);
                break;
            default:
                LogUtil.e(TAG, "lrn type illegal:" + normRegion);
                break;
        }
    }
}

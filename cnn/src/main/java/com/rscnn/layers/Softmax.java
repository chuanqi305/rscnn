package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Softmax;

public class Softmax extends Layer {
    private int ignoreLabel;            // currently not used
    private boolean normalize;          // currently not used

    private static final String TAG = "Softmax";

    private Allocation expAlloc;
    private Allocation expSumAlloc;
    private ScriptC_Softmax softmaxScript;

    public void setIgnoreLabel(int ignoreLabel) {
        this.ignoreLabel = ignoreLabel;
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    public Softmax() {
    }

    @Override
    public void setup(){

        int n = inputShape[0][0];
        int h = inputShape[0][1];
        int w = inputShape[0][2];
        int c = inputShape[0][3];

        Type expSumType = Type.createX(renderScript, Element.F32(renderScript), n * h * w);
        expSumAlloc = Allocation.createTyped(renderScript, expSumType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        softmaxScript = new ScriptC_Softmax(renderScript);
        softmaxScript.set_channel(c);
        softmaxScript.set_expSum(expSumAlloc);
    }

    @Override
    public void computeFeatureMap() {
        int n = inputShape[0][0];
        int h = inputShape[0][1];
        int w = inputShape[0][2];
        int c = inputShape[0][3];

        int channelAligned = getInputChannelAligned();

        FeatureMap input = (FeatureMap) featureMapInput[0];

        Allocation inAllocation = input.getFeatureMap();

        if(inAllocation.getElement().getVectorSize()==4){
            softmaxScript.set_channelAligned(channelAligned);
        }
        else{
            softmaxScript.set_channelAligned(c);
        }

        softmaxScript.set_inBlob(inAllocation);
        Script.LaunchOptions option = new Script.LaunchOptions();
        option.setX(0, n * h * w * c);
        softmaxScript.forEach_compute_exp(option);
        option.setX(0, n * h * w);
        softmaxScript.forEach_compute_exp_sum(option);
        option.setX(0, n * h * w * c);
        softmaxScript.forEach_compute(option);
        featureMapOutput = featureMapInput[0];
    }

    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
        if(softmaxScript!=null){
            int n = inputShape[0][0];
            int h = inputShape[0][1];
            int w = inputShape[0][2];
            int c = inputShape[0][3];
            Type expSumType = Type.createX(renderScript, Element.F32(renderScript), n * h * w);
            expSumAlloc = Allocation.createTyped(renderScript, expSumType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
            softmaxScript.set_channel(c);
            softmaxScript.set_expSum(expSumAlloc);
        }
    }
}
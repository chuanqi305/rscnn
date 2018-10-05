package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Dropout;

/**
 * Dropout operation is in-place.
 */
public class Dropout extends Layer {
    private float dropoutRatio = 0.5f;
    boolean scaleTrain = false;
    private ScriptC_Dropout scriptDropout;

    public void setScaleTrain(boolean scaleTrain) {
        this.scaleTrain = scaleTrain;
    }

    public void setDropoutRatio(float dropoutRatio) {
        this.dropoutRatio = dropoutRatio;
    }

    public Dropout() {
    }

    @Override
    public void setup()
    {
        scriptDropout = new ScriptC_Dropout(renderScript);
        scriptDropout.set_scale_ratio(1 - dropoutRatio);
    }

    @Override
    public void computeFeatureMap()
    {
        FeatureMap input = (FeatureMap)featureMapInput[0];
        Allocation frameAllocation = input.getFeatureMap();
        if (!input.isPad4()) {
            Script.LaunchOptions options = new Script.LaunchOptions();
            options.setX(0, getOutputChannelAligned() / 4).setY(0, outputShape[1] * outputShape[2]);
            scriptDropout.set_InputData(frameAllocation);
            scriptDropout.set_OutputData(frameAllocation);
            scriptDropout.forEach_compute_vector4(options);
        } else {
            scriptDropout.forEach_compute(frameAllocation, frameAllocation);
        }

        featureMapOutput = featureMapInput[0];
    }

    @Override
    public void computeOutputShape()
    {
        outputShape = inputShape[0];
    }
}

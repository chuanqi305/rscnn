package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Flatten;

public class Flatten extends Layer {
    private ScriptC_Flatten scriptCFlatten;
    @Override
    public void computeFeatureMap() {
        FeatureMap input = (FeatureMap) featureMapInput[0];
        FeatureMap output = (FeatureMap) featureMapOutput;
        scriptCFlatten.set_InputData(input.getFeatureMap());
        scriptCFlatten.set_OutputData(output.getFeatureMap());
        Allocation frameAllocation = input.getFeatureMap();
        if(!input.isMatrix2D()){
            output.setFeatureMap(input.getFeatureMap());
            return;
        }
        if (inputShape[0][3] % 4 == 0) {
            Script.LaunchOptions options = new Script.LaunchOptions();
            options.setX(0, getInputChannelAligned() / 4)
                    .setY(0, inputShape[0][1] * inputShape[0][2]);
            scriptCFlatten.forEach_compute_vector4(options);
        } else {
            scriptCFlatten.forEach_compute(frameAllocation);
        }
    }

    @Override
    public void computeOutputShape() {
        int n = inputShape[0][0];
        int h = inputShape[0][1];
        int w = inputShape[0][2];
        int c = inputShape[0][3];
        outputShape = new int[]{n, 1, 1, h * w * c};
        scriptCFlatten = new ScriptC_Flatten(renderScript);
        scriptCFlatten.set_channels(c);
        scriptCFlatten.set_blocks(getInputChannelAligned() / 4);
        allocFeatureMapNoPad();
    }
}

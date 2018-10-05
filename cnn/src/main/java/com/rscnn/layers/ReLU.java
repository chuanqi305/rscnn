package com.rscnn.layers;

import android.renderscript.Allocation;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_ReLU;

/**
 * ReLU operation can be in-place
 */
public class ReLU extends Layer {

    private static final String TAG = "ReLU";

    private ScriptC_ReLU scriptRelu;

    private boolean computed = false;

    public void setComputed(boolean computed) {
        this.computed = computed;
    }

    public ReLU() {
    }

    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
    }

    @Override
    public void setup()
    {
        scriptRelu = new ScriptC_ReLU(renderScript);
    }

    @Override
    public void computeFeatureMap()
    {
        FeatureMap input = (FeatureMap) featureMapInput[0];
        if(!computed) {
            Allocation frameAllocation = input.getFeatureMap();
            if (input.isPad4()) {
                scriptRelu.set_InputData(input.getFeatureMap());
                scriptRelu.set_OutputData(input.getFeatureMap());
                scriptRelu.forEach_compute_vector4(getLaunchOptionVector4());
            } else {
                scriptRelu.forEach_compute(frameAllocation, frameAllocation);
            }
        }
        featureMapOutput = input;
    }
}

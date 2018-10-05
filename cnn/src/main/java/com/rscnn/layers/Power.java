package com.rscnn.layers;

import android.renderscript.Allocation;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Power;

public class Power extends Layer {
    private float power = 1.f;
    private float shift = 1.f;
    private float scale = 0.f;
    private ScriptC_Power scriptCPower;

    public void setPower(float power) {
        this.power = power;
    }

    public void setShift(float shift) {
        this.shift = shift;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }

    @Override
    public void setup()
    {
        scriptCPower = new ScriptC_Power(renderScript);
        scriptCPower.set_power(power);
        scriptCPower.set_scale(scale);
        scriptCPower.set_shift(shift);
    }

    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
    }

    @Override
    public void computeFeatureMap(){
        FeatureMap input = (FeatureMap)featureMapInput[0];
        Allocation frameAllocation = input.getFeatureMap();
        if (input.isPad4()) {
            scriptCPower.set_InputData(frameAllocation);
            scriptCPower.forEach_compute_vector4(getLaunchOptionVector4());
        } else {
            scriptCPower.forEach_compute(frameAllocation, frameAllocation);
        }
        featureMapOutput = featureMapInput[0];
    }
}

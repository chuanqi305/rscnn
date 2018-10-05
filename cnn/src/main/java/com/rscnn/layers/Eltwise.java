package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_Eltwise;

public class Eltwise extends Layer {

    private String operation = "SUM";
    private float[] coeff;
    private ScriptC_Eltwise scriptEltwise;

    public void setOperation(String operation) {
        this.operation = operation;
    }

    public void setCoeff(float[] coeff) {
        this.coeff = coeff;
    }

    @Override
    public void setup(){
        scriptEltwise = new ScriptC_Eltwise(renderScript);
        allocFeatureMap();
    }

    @Override
    public void computeFeatureMap() {
        FeatureMap output = (FeatureMap) featureMapOutput;
        Allocation outAllocation = output.getFeatureMap();
        scriptEltwise.set_OutputData(outAllocation);
        Script.LaunchOptions options = new Script.LaunchOptions();
        int outputChannelAligned = getOutputChannelAligned();
        options.setX(0, outputChannelAligned / 4).setY(0, outputShape[1] * outputShape[2]);

        for(int i = 0; i< featureMapInput.length; i++){
            FeatureMap input = (FeatureMap) featureMapInput[i];
            Allocation frameAllocation = input.getFeatureMap();
            scriptEltwise.set_InputData(frameAllocation);
            switch (operation){
                case "SUM":
                    float coeffFactor = 1.f;
                    if(coeff!=null && i < coeff.length){
                        coeffFactor = coeff[i];
                    }
                    if(i==0){
                        scriptEltwise.forEach_set_zero_vector4(options);
                    }
                    scriptEltwise.set_coeff(coeffFactor);
                    scriptEltwise.forEach_compute_sum_vector4(options);
                    break;
                case "MAX":
                    if(i==0){
                        scriptEltwise.forEach_copy_vector4(options);
                        continue;
                    }
                    scriptEltwise.forEach_compute_max_vector4(options);
                    break;
                case "PROD":
                    if(i==0){
                        scriptEltwise.forEach_copy_vector4(options);
                        continue;
                    }
                    scriptEltwise.forEach_compute_mul_vector4(options);
                    break;
                default:
                    LogUtil.e("Eltwise", "Eltwise type "+operation+" has not implemented");
                    break;
            }
        }
    }
    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
    }
}

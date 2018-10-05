package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.network.LayerParamInterface;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_BatchNorm;

public class BatchNorm extends Layer implements LayerParamInterface {
    private float epsilon = 1e-5f;
    private float movingAverageFraction = 0.999f;
    private boolean useGlobalStats = true;
    private final boolean paralell = false;

    private float[] mean;
    private float[] variance;
    private float scale;

    private ScriptC_BatchNorm scriptBatchNorm;


    private float[] revStandard;// compute 1/sqrt(variance - epsilon) for cache

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    public void setMovingAverageFraction(float movingAverageFraction) {
        this.movingAverageFraction = movingAverageFraction;
    }

    public void setUseGlobalStats(boolean useGlobalStats) {
        this.useGlobalStats = useGlobalStats;
    }

    @Override
    public void loadParams(byte[][] params) {
        LogUtil.e("BatchNorm", "fastLoadParams is not implemented");
    }

    @Override
    public void setup(){
        int channel = outputShape[3];
        int channelAlign = channel;
        if(channel % 4 != 0){
            channelAlign = channel + 4 - channel % 4;
        }

        float scaleFactor = scale == 0 ? 0 : 1.f / scale;
        for (int i = 0; i < mean.length; i++) {
            mean[i] *= scaleFactor;
            variance[i] *= scaleFactor;
        }

        this.revStandard = new float[mean.length];//compute something for cache
        for (int i = 0; i < mean.length; i++) {
            float std = (float)Math.sqrt(variance[i] + epsilon);
            revStandard[i] = 1.0f / std;
        }
        scriptBatchNorm = new ScriptC_BatchNorm(renderScript);

        float[] meanArray = new float[channelAlign];
        float[] rstdArray = new float[channelAlign];

        for(int i=0;i<channel;i++){
            meanArray[i] = mean[i];
            rstdArray[i] = revStandard[i];
        }

        Allocation meanAllocation;
        Allocation rstdAllocation;
        Type scaleType = Type.createX(renderScript, Element.F32_4(renderScript), channelAlign / 4);
        Type biasType = Type.createX(renderScript, Element.F32_4(renderScript), channelAlign / 4);

        meanAllocation = Allocation.createTyped(renderScript, scaleType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        meanAllocation.copyFrom(meanArray);

        rstdAllocation = Allocation.createTyped(renderScript, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        rstdAllocation.copyFrom(rstdArray);

        scriptBatchNorm.set_mean_blob(meanAllocation);
        scriptBatchNorm.set_reverse_std_blob(rstdAllocation);
    }

    @Override
    public void computeOutputShape()
    {
        outputShape = inputShape[0];
    }

    @Override
    public void computeFeatureMap()
    {
        FeatureMap input = (FeatureMap)featureMapInput[0];
        Allocation frameAllocation = input.getFeatureMap();

        if(input.isPad4()) {
            Script.LaunchOptions option = new Script.LaunchOptions();
            option.setX(0, getOutputChannelAligned() / 4).setY(0, outputShape[1] * outputShape[2]);
            scriptBatchNorm.set_InputData(frameAllocation);
            scriptBatchNorm.forEach_compute_vector4(option);
        }
        else {
            scriptBatchNorm.forEach_compute(frameAllocation, frameAllocation);
        }
        featureMapOutput = featureMapInput[0];
    }
}

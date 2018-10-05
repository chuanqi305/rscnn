package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.network.LayerParamInterface;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_Scale;

public class Scale extends Layer implements LayerParamInterface {
    private int axis = 1;
    private int numAxes = 1;
    private boolean biasTerm = false;
    private float[] scale;
    private float[] bias;
    private ScriptC_Scale scriptScale;

    public void setAxis(int axis) {
        this.axis = axis;
    }

    public void setNumAxes(int numAxes) {
        this.numAxes = numAxes;
    }

    public void setBiasTerm(boolean biasTerm) {
        this.biasTerm = biasTerm;
    }

    @Override
    public void loadParams(byte[][] params) {
        LogUtil.e("Scale", "loadParams is not implemented");
    }

    @Override
    public void computeOutputShape()
    {
        outputShape = inputShape[0];
    }

    @Override
    public void setup()
    {
        int channel = outputShape[3];
        int channelAlign = channel;
        if(channel % 4 != 0){
            channelAlign = channel + 4 - channel % 4;
        }

        scriptScale = new ScriptC_Scale(renderScript);

        float[] scaleArray = new float[channelAlign];
        float[] biasArray = new float[channelAlign];

        for(int i=0;i<channel;i++){
            scaleArray[i] = scale[i];
            biasArray[i] = bias[i];
        }

        Allocation scaleAllocation;
        Allocation biasAllocation;
        Type scaleType = Type.createX(renderScript, Element.F32_4(renderScript), channelAlign / 4);
        Type biasType = Type.createX(renderScript, Element.F32_4(renderScript), channelAlign / 4);

        scaleAllocation = Allocation.createTyped(renderScript, scaleType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        scaleAllocation.copyFrom(scaleArray);

        biasAllocation = Allocation.createTyped(renderScript, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        scriptScale.set_scale(scaleAllocation);
        scriptScale.set_bias(biasAllocation);

    }

    @Override
    public void computeFeatureMap()
    {
        Allocation frameAllocation = ((FeatureMap)featureMapInput[0]).getFeatureMap();
        Script.LaunchOptions option = new Script.LaunchOptions();
        option.setX(0, getOutputChannelAligned() / 4).setY(0, outputShape[1] * outputShape[2]);
        scriptScale.set_InputData(frameAllocation);
        scriptScale.forEach_compute_vector4(option);
        featureMapOutput = featureMapInput[0];
    }
}

package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_PSROIPooling;

public class PSROIPooling extends Layer {
    private float spatialScale = 1.f;
    private int outputDim = 2; // output channel number
    private int groupSize = 3; // equal to pooled_size
    private int pooledH;
    private int pooledW;

    ScriptC_PSROIPooling psroiPooling;

    public void setSpatialScale(float spatialScale) {
        this.spatialScale = spatialScale;
    }

    public void setOutputDim(int outputDim) {
        this.outputDim = outputDim;
    }

    public void setGroupSize(int groupSize) {
        this.groupSize = groupSize;
    }

    @Override
    public void setup(){
        int height = inputShape[0][1];
        int width = inputShape[0][2];
        int channels = inputShape[0][3];
        psroiPooling = new ScriptC_PSROIPooling(renderScript);
        psroiPooling.set_outputDim(outputDim);
        psroiPooling.set_groupSize(groupSize);
        psroiPooling.set_spatialScale(spatialScale);

        psroiPooling.set_width(width);
        psroiPooling.set_height(height);
        psroiPooling.set_channels(channels);
    }


    @Override
    public void computeFeatureMap(){
        int count = inputShape[0][0];
        Allocation bottomData = ((FeatureMap)featureMapInput[0]).getFeatureMap();
        Allocation bottomRois = ((FeatureMap)featureMapInput[0]).getFeatureMap();
        Allocation topData = ((FeatureMap)featureMapInput[0]).getFeatureMap();
        psroiPooling.set_bottom_data(bottomData);
        psroiPooling.set_bottom_rois(bottomRois);
        psroiPooling.set_top_data(topData);
        Script.LaunchOptions option = new Script.LaunchOptions();
        option.setX(0, pooledH * pooledW * outputDim);
        psroiPooling.forEach_compute(option);

    }

    @Override
    public void computeOutputShape(){
        pooledH = groupSize;
        pooledW = groupSize;
        outputShape = new int[]{inputShape[1][2], pooledH, pooledW, outputDim};
        allocFeatureMapBlock4();
    }
}

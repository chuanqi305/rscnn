package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_ROIPooling;

public class ROIPooling extends Layer {

    private int pooledW = 6;
    private int pooledH = 6;
    private float spatialScale = 0.0625f;
    private ScriptC_ROIPooling roiPoolScript;

    public void setPooledW(int pooledW) {
        this.pooledW = pooledW;
    }

    public void setPooledH(int pooledH) {
        this.pooledH = pooledH;
    }

    public void setSpatialScale(float spatialScale) {
        this.spatialScale = spatialScale;
    }

    @Override
    public void setup(){
        roiPoolScript = new ScriptC_ROIPooling(renderScript);
        roiPoolScript.set_pooled_height(pooledH);
        roiPoolScript.set_pooled_width(pooledW);
        roiPoolScript.set_pool_area(pooledH * pooledW);
        roiPoolScript.set_spatial_scale(spatialScale);

        int featureMapHeight = inputShape[0][1];
        int featureMapWidth = inputShape[0][2];
        int channelCount = inputShape[0][3];

        int roiCount = inputShape[1][0];

        roiPoolScript.set_fmap_width(featureMapWidth);
        roiPoolScript.set_fmap_height(featureMapHeight);
        roiPoolScript.set_channel_count(channelCount);
        roiPoolScript.set_roi_count(roiCount);
        roiPoolScript.set_batch_pool_area(pooledH * pooledW * channelCount);
        roiPoolScript.set_fmap_area(featureMapHeight * featureMapWidth);
    }

    public ROIPooling() {
    }

    @Override
    public void computeFeatureMap() {
        FeatureMap fm = (FeatureMap) featureMapInput[0];
        Allocation fmInput = fm.getFeatureMap();
        FeatureMap roi = (FeatureMap) featureMapInput[1];
        Allocation roiInput = roi.getFeatureMap();

        FeatureMap output = (FeatureMap) featureMapOutput;
        Allocation outAllocation = output.getFeatureMap();

        roiPoolScript.set_fmap_blob(fmInput);
        roiPoolScript.set_roi_blob(roiInput);
        roiPoolScript.set_out_blob(outAllocation);
        int roiCount = inputShape[1][0];
        int channel = inputShape[1][3];
        if(channel==256){
            Script.LaunchOptions option = new Script.LaunchOptions();
            option.setX(0, roiCount * pooledH * pooledW);
            roiPoolScript.forEach_compute_channel256(option);
        }
        else {
            roiPoolScript.forEach_compute_vector(outAllocation);
        }
    }

    @Override
    public void computeOutputShape() {
        int roiCount = inputShape[1][2];// proposal output [1][1][200][4]
        int channelCount = inputShape[0][3];
        outputShape = new int[]{roiCount, pooledH, pooledW, channelCount};
        allocFeatureMapBlock4();
        if(roiPoolScript!=null){
            int featureMapHeight = inputShape[0][1];
            int featureMapWidth = inputShape[0][2];
            roiPoolScript.set_fmap_width(featureMapWidth);
            roiPoolScript.set_fmap_height(featureMapHeight);
        }
    }
}

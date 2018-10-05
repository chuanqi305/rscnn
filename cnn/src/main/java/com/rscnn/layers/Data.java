package com.rscnn.layers;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

public class Data extends Layer {
    private int[] dim;
    public void setDim(int[] dim) {
        this.dim = dim;
    }

    public int[] getDim() {
        return dim;
    }

    private float[] padChannel4(float[] input){
        int len = input.length;
        int blocks = len / 3;
        int newLen = blocks * 4;
        float[] output = new float[newLen];
        for(int i=0; i< blocks;i++){
            output[i * 4] = input[i * 3];
            output[i * 4 + 1] = input[i * 3 + 1];
            output[i * 4 + 2] = input[i * 3 + 2];
            output[i * 4 + 3] = 0.f;
        }
        return output;
    }

    @Override
    public void computeFeatureMap(){
        float[] bmp = (float[])featureMapInput[0];
        FeatureMap featureMap = (FeatureMap) featureMapOutput;
        featureMap.getFeatureMap().copyFrom(bmp);
    }

    @Override
    public void computeOutputShape(){
        outputShape =  inputShape[0];
        allocFeatureMap();
    }
}

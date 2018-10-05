package com.rscnn.layers;

import com.rscnn.network.Layer;

public class ImInfo  extends Layer {
    private int[] dim;

    public void setDim(int[] dim) {
        this.dim = dim;
    }

    @Override
    public void computeFeatureMap()
    {
        featureMapOutput = featureMapInput[1];
    }

    @Override
    public void computeOutputShape(){
        outputShape =  new int[]{3};
    }
}

package com.rscnn.layers;

import com.rscnn.network.Layer;

public class Permute extends Layer {
    private int[] order;

    public void setOrder(int[] order) {
        this.order = order;
    }

    @Override
    public void computeFeatureMap() {
        featureMapOutput = featureMapInput[0];// use N * H * W * C internal, need not permute
    }

    @Override
    public void computeOutputShape() {
        outputShape = inputShape[0];
    }
}

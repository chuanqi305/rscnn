package com.rscnn.layers;

import com.rscnn.network.Layer;

public final class Input extends Layer {
    private int[] dim = new int[4];// not used
    private int[] meanValue;// not used

    private int inputHeight = 128;
    private int inputWidth = 128;

    public void setDim(int[] dim) {
        this.dim = dim;
        this.inputHeight = dim[2];
        this.inputWidth = dim[3];
    }

    public void setCropSize(int cropSize) {
        dim[2] = dim[3] = cropSize;
    }

    public void setMeanValue(int[] meanValue) {
        this.meanValue = meanValue;
    }

    public void setInputHeight(int inputHeight) {
        this.inputHeight = inputHeight;
        this.dim[2] = inputHeight;
    }

    public void setInputWidth(int inputWidth) {
        this.inputWidth = inputWidth;
        this.dim[3] = inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int[] getDim() {
        return dim;
    }

    @Override
    public void computeFeatureMap() {
        featureMapOutput = new Object[featureMapInput.length];
        System.arraycopy(featureMapInput, 0, featureMapOutput, 0, featureMapInput.length);
    }

    @Override
    public void computeOutputShape(){
        inputShape = new int[][]{{inputHeight, inputWidth}};
        outputShape = new int[]{1, inputHeight, inputWidth, 3};
    }
}

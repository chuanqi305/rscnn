package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Pooling;

public class Pooling extends Layer {
    private String kernelType = "MAX";
    private int padH = 0;
    private int padW = 0;
    private int strideH = 1;
    private int strideW = 1;
    private int kernelH;
    private int kernelW;
    private boolean globalPooling = false;

    private ScriptC_Pooling scriptPooling = null;

    private static final String TAG = "Pooling";

    public void setPadH(int padH) {
        this.padH = padH;
    }

    public void setPadW(int padW) {
        this.padW = padW;
    }

    public void setStrideH(int strideH) {
        this.strideH = strideH;
    }

    public void setStrideW(int strideW) {
        this.strideW = strideW;
    }

    public void setKernelH(int kernelH) {
        this.kernelH = kernelH;
    }

    public void setKernelW(int kernelW) {
        this.kernelW = kernelW;
    }

    public void setGlobalPooling(boolean globalPooling) {
        this.globalPooling = globalPooling;
    }

    public void setKernelSize(int kernelSize) {
        this.kernelH = this.kernelW = kernelSize;
    }
    public void setPad(int pad) {
        this.padH = this.padW = pad;
    }
    public void setStride(int stride) {
        this.strideH = this.strideW = stride;
    }
    public void setPool(String pool){// max,ave
        this.kernelType = pool.toLowerCase();
    }

    public Pooling() {
    }

    @Override
    public void setup(){
        scriptPooling = new ScriptC_Pooling(renderScript);
        scriptPooling.set_kernel_h(kernelH);
        scriptPooling.set_kernel_w(kernelW);
        scriptPooling.set_pad_h(padH);
        scriptPooling.set_pad_w(padW);
        scriptPooling.set_stride_h(strideH);
        scriptPooling.set_stride_w(strideW);
        scriptPooling.set_kernel_area(kernelH * kernelW);

        int h_i = inputShape[0][1];
        int w_i = inputShape[0][2];

        int outHeight = outputShape[1];
        int outWidth = outputShape[2];
        int outChannel = outputShape[3];
        int inChannel = inputShape[0][3];

        int outChannelAlign = outChannel;
        if(outChannelAlign % 4 !=0){
            outChannelAlign = outChannel + 4 - outChannel % 4;
        }
        scriptPooling.set_out_channel_aligned(outChannelAlign);
        scriptPooling.set_in_channel(inChannel);
        scriptPooling.set_width(w_i);
        scriptPooling.set_height(h_i);
        scriptPooling.set_feature_map_size(w_i * h_i);
        scriptPooling.set_pool_height(outHeight);
        scriptPooling.set_pool_width(outWidth);
        scriptPooling.set_pool_size(outHeight * outWidth);
    }

    @Override
    public void computeFeatureMap() {
        FeatureMap input = (FeatureMap) featureMapInput[0];
        Allocation frameAllocation = input.getFeatureMap();
        FeatureMap output = (FeatureMap) featureMapOutput;
        Allocation outAllocation = output.getFeatureMap();
        scriptPooling.set_in_blob(frameAllocation);
        scriptPooling.set_out_blob(outAllocation);
        int outputChannelAligned = getOutputChannelAligned();
        Script.LaunchOptions options = new Script.LaunchOptions();
        options.setX(0, outputChannelAligned / 4).setY(0, outputShape[1] * outputShape[2]);

        if(globalPooling){
            options.setX(0, outputChannelAligned / 4).setY(0, 1);
            scriptPooling.forEach_global_pooling_2d(options);
            float[][][][] data = input.getData();
            float[][][][] data2 = output.getData();
            int len = data.length;
        }
        else if(kernelType.equalsIgnoreCase("MAX")) {
            scriptPooling.forEach_max_pooling_2d(options);
        }
        else if(kernelType.equalsIgnoreCase("AVE")){
            scriptPooling.forEach_mean_pooling_2d(options);
        }
    }

    @Override
    public void computeOutputShape(){
        int inNum = inputShape[0][0];
        int inHeight = inputShape[0][1];
        int inWidth = inputShape[0][2];
        int inChannel = inputShape[0][3];
        int outHeight;
        int outWidth;

        if(Layer.weightFromTensorFlow){
            outHeight = (int) ((inHeight + 2 * padH - kernelH) / (double) strideH + 1);
            outWidth = (int) ((inWidth + 2 * padW - kernelW) / (double) strideW + 1);
        }
        else{
            outHeight = (int) (Math.ceil((inHeight + 2 * padH - kernelH) / (double) strideH) + 1);
            outWidth = (int) (Math.ceil((inWidth + 2 * padW - kernelW) / (double) strideW) + 1);
        }

        if(globalPooling){
            outHeight = 1;
            outWidth = 1;
        }

        outputShape = new int[]{inNum, outHeight, outWidth, inChannel};
        allocFeatureMap();

        if(scriptPooling!=null){
            scriptPooling.set_width(inWidth);
            scriptPooling.set_height(inHeight);
            scriptPooling.set_feature_map_size(inWidth * inHeight);
            scriptPooling.set_pool_height(outHeight);
            scriptPooling.set_pool_width(outWidth);
            scriptPooling.set_pool_size(outHeight * outWidth);
        }
    }
}
package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.ScriptC;
import android.renderscript.ScriptIntrinsicBLAS;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.network.LayerParamInterface;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_Convolution;

public class Convolution extends Layer implements LayerParamInterface {

    private int numOutput;
    private boolean biasTerm = true;

    private int padH;
    private int padW;
    private int kernelH;
    private int kernelW;
    private int strideH = 1;
    private int strideW = 1;
    private int group = 1;
    private int dilation = 1;

    private float[][][][] weight;
    private float[] bias;

    private ScriptC_Convolution scriptConvolution;
    Allocation kernelAllocation;
    Allocation biasAllocation;

    private boolean nextRelu = false;

    private byte[] kernelBuffer;
    private byte[] biasBuffer;

    private static final String TAG = "Convolution";

    private boolean conv1x1UserIntrinsic = false; // some phone not support input transpose.
    private boolean convnxnUseIntrinsic = false; // intrinsic is slower.

    public void setKernelSize(int kernelSize) {
        this.kernelH = this.kernelW = kernelSize;
    }

    public void setPad(int pad) {
        this.padH = this.padW = pad;
    }

    public void setStride(int stride) {
        this.strideH = this.strideW = stride;
    }

    public void setNumOutput(int numOutput) {
        this.numOutput = numOutput;
    }

    public void setBiasTerm(boolean biasTerm) {
        this.biasTerm = biasTerm;
    }

    public void setPadH(int padH) {
        this.padH = padH;
    }

    public void setPadW(int padW) {
        this.padW = padW;
    }

    public void setKernelH(int kernelH) {
        this.kernelH = kernelH;
    }

    public void setKernelW(int kernelW) {
        this.kernelW = kernelW;
    }

    public void setStrideH(int strideH) {
        this.strideH = strideH;
    }

    public void setStrideW(int strideW) {
        this.strideW = strideW;
    }

    public void setDilation(int dilation) {
        this.dilation = dilation;
    }

    public void setGroup(int group) {
        this.group = group;
    }

    public boolean isNextRelu() {
        return nextRelu;
    }

    public void setNextRelu(boolean nextRelu) {
        this.nextRelu = nextRelu;
    }

    @Override
    public void loadParams(byte[][] params) {
        kernelBuffer = params[0];
        if(biasTerm){
            biasBuffer = params[1];
        }
    }

    public Convolution() {

    }

    @Override
    public void setup()
    {
        initKernel();
    }

    @Override
    public void computeFeatureMap() {
        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];
        int inputChannel = inputShape[0][3];
        int outputChannel = outputShape[3];

        int outputChannelAligned = getOutputChannelAligned();
        int inputChannelAligned = getInputChannelAligned();

        FeatureMap input = (FeatureMap) featureMapInput[0];
        FeatureMap output = (FeatureMap) featureMapOutput;

        Allocation inputFeatureMap = input.getFeatureMap();
        Allocation outputFeatureMap = output.getFeatureMap();

        scriptConvolution.set_InputData(inputFeatureMap);
        scriptConvolution.set_OutputData(outputFeatureMap);
        ScriptC.LaunchOptions option = new Script.LaunchOptions();
        boolean useIntrinsicBlas = false;

        if(conv1x1UserIntrinsic && kernelH==1 && kernelW==1){
            useIntrinsicBlas = true;
        }
        if(convnxnUseIntrinsic && kernelH!=1 && kernelW!=1){
            conv1x1UserIntrinsic = true;
        }

        if(useIntrinsicBlas){
            if(kernelH==1 && kernelW==1){
                scriptIntrinsicBLAS.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.TRANSPOSE,
                        1.f, inputFeatureMap, kernelAllocation, 0.f, outputFeatureMap);
            }
            else if (inputChannel == group) {
                option.setX(0, getInputChannelAligned() / 4).setY(0, outputHeight * outputWidth);
                scriptConvolution.forEach_conv_dw4(option);
                return;
            }
            else {
                Type.Builder colType = new Type.Builder(renderScript, Element.F32(renderScript));
                colType.setX(kernelH * kernelW * inputChannelAligned).setY(outputHeight * outputWidth);
                Allocation colAllocation = Allocation.createTyped(renderScript, colType.create());
                scriptConvolution.set_ColData(colAllocation);

                option.setX(0, kernelH * kernelW).setY(0, outputHeight * outputWidth);
                scriptConvolution.forEach_conv_im2col2(option);

                scriptIntrinsicBLAS.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.TRANSPOSE,
                        1.f, colAllocation, kernelAllocation, 0.f, outputFeatureMap);

                colAllocation.destroy();
            }

            if(nextRelu && biasTerm){
                scriptConvolution.forEach_conv_bias_relu(outputFeatureMap, outputFeatureMap);
            }
            else if(biasTerm){
                scriptConvolution.forEach_conv_bias(outputFeatureMap, outputFeatureMap);
            }
            else if(nextRelu){
                scriptConvolution.forEach_conv_relu(outputFeatureMap, outputFeatureMap);
            }
        }
        else {
            if(kernelH==1 && kernelW==1){
                option.setX(0, getOutputChannelAligned() / 4).setY(0, outputHeight * outputWidth);
                scriptConvolution.forEach_conv1x1(option);
            }
            else if (inputChannel == group) {
                option.setX(0, getInputChannelAligned() / 4).setY(0, outputHeight * outputWidth);
                scriptConvolution.forEach_conv_dw4(option);
            }
            else {
                int blockSize = 4;
                int[] blockSizeList = {256, 128, 96, 64, 48, 32, 16, 8};
                for (int blk : blockSizeList) {
                    if (outputChannelAligned % blk == 0) {
                        blockSize = blk;
                        break;
                    }
                }
                scriptConvolution.set_nblock(blockSize);
                option.setX(0, outputChannelAligned / blockSize).setY(0, outputHeight * outputWidth);
                scriptConvolution.forEach_conv4n(option);
            }
        }
    }

    private void initKernel() {
        int outputChannel = outputShape[3];//100
        int outputChannelAligned = getOutputChannelAligned();
        int inputChannel = inputShape[0][3]; //256
        int kernelHeight = kernelH;
        int kernelWidth = kernelW;
        int kernelSize = kernelHeight * kernelWidth;

        int inputChannelGroup = getInputChannelAligned() / group;

        int totalKernelSize = inputChannelGroup * outputChannelAligned * kernelSize;

        int inHeight = inputShape[0][1];
        int inWidth = inputShape[0][2];

        int outHeight = outputShape[1];
        int outWidth = outputShape[2];

        Type.Builder kernelType = new Type.Builder(renderScript, Element.F32(renderScript));
        kernelType.setY(outputChannelAligned);
        kernelType.setX(kernelH * kernelW * inputChannelGroup);
        Type biasType = Type.createX(renderScript, Element.F32(renderScript), outputChannelAligned);

        kernelAllocation = Allocation.createTyped(renderScript, kernelType.create());
        biasAllocation = Allocation.createTyped(renderScript, biasType);

        if(kernelBuffer==null) {
            float[] kernelMatrix = new float[totalKernelSize];
            float[] biasMatrix = new float[outputChannelAligned];

            int count = 0;

            //kernel -> [o][h][w][i]
            for (int i = 0; i < outputChannelAligned; i++) {
                for (int k = 0; k < kernelHeight; k++) {
                    for (int l = 0; l < kernelWidth; l++) {
                        for (int j = 0; j < inputChannelGroup; j++) {
                            if (i >= outputChannel || j >= inputChannel) {
                                kernelMatrix[count++] = 0;
                            } else {
                                kernelMatrix[count++] = weight[i][j][k][l];
                            }
                        }
                    }
                }
            }

            count = 0;
            for (int i = 0; i < outputChannelAligned; i++) {
                if (i >= outputChannel) {
                    biasMatrix[count++] = 0;
                } else {
                    biasMatrix[count++] = bias[i];
                }
            }
            kernelAllocation.copyFrom(kernelMatrix);
            biasAllocation.copyFrom(biasMatrix);
        }
        else{
            kernelAllocation.copyFromUnchecked(kernelBuffer);
            if(biasBuffer!=null) {
                biasAllocation.copyFromUnchecked(biasBuffer);
            }
            else{
                biasAllocation.copyFrom(new float[getOutputChannelAligned()]);
            }
        }
        scriptConvolution = new ScriptC_Convolution(renderScript);
        scriptConvolution.set_KernelData(kernelAllocation);
        scriptConvolution.set_BiasData(biasAllocation);

        scriptConvolution.set_inputHeight(inHeight);
        scriptConvolution.set_inputWidth(inWidth);
        scriptConvolution.set_outputHeight(outHeight);
        scriptConvolution.set_outputWidth(outWidth);

        scriptConvolution.set_inputChannel(inputChannel);
        scriptConvolution.set_inputChannelAligned(inputChannelGroup);

        scriptConvolution.set_kernelH(kernelHeight);
        scriptConvolution.set_kernelW(kernelWidth);
        scriptConvolution.set_padH(padH);
        scriptConvolution.set_padW(padW);
        scriptConvolution.set_strideH(strideH);
        scriptConvolution.set_strideW(strideW);

        scriptConvolution.set_kernelSize(kernelHeight * kernelWidth);
        scriptConvolution.set_group(group);
        scriptConvolution.set_dilation(dilation);

        if(nextRelu){
            scriptConvolution.set_relu(1);
        }
        else{
            scriptConvolution.set_relu(0);
        }

        scriptIntrinsicBLAS = ScriptIntrinsicBLAS.create(renderScript);

        weight = null;
        bias = null;
    }

    @Override
    public void computeOutputShape(){
        int inNum = inputShape[0][0];
        int inHeight = inputShape[0][1];
        int inWidth = inputShape[0][2];

        int kernelNum = numOutput;
        int kernelHeight = kernelH;
        int kernelWidth = kernelW;

        int kernelExtentH = dilation * (kernelHeight - 1) + 1;
        int kernelExtentW = dilation * (kernelWidth - 1) + 1;

        int outHeight = (inHeight + 2 * padH - kernelExtentH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelExtentW) / strideW + 1;

        int outNum = inNum;

        outputShape = new int[]{outNum, outHeight, outWidth, kernelNum};

        allocFeatureMap();
        if(scriptConvolution!=null){// resize input
            scriptConvolution.set_inputHeight(inHeight);
            scriptConvolution.set_inputWidth(inWidth);
            scriptConvolution.set_outputHeight(outHeight);
            scriptConvolution.set_outputWidth(outWidth);
        }
    }
}


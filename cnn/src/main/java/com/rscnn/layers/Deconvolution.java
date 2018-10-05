package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.network.LayerParamInterface;
import com.rscnn.utils.DebugUtil;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_Deconvolution;

public class Deconvolution extends Layer implements LayerParamInterface {

    private int numOutput;
    private boolean biasTerm = true;
    private int padH;
    private int padW;
    private int kernelH;
    private int kernelW;
    private int strideH;
    private int strideW;
    private int group;

    private float[][][][] weight;
    private float[] bias;

    private byte[] weightBuffer;
    private byte[] biasBuffer;
    
    private ScriptC_Deconvolution scriptDeconvolution;

    public void setNumOutput(int numOutput) {
        this.numOutput = numOutput;
    }

    public void setBiasTerm(boolean biasTerm) {
        this.biasTerm = biasTerm;
    }

    public void setPad(int pad) {
        this.padH = this.padW = pad;
    }

    public void setKernelSize(int kernelSize) {
        this.kernelH = this.kernelW = kernelSize;
    }

    public void setStride(int stride) {
        this.strideH = this.strideW = stride;
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

    public void setGroup(int group) {
        this.group = group;
    }

    @Override
    public void loadParams(byte[][] params) {
        weightBuffer = params[0];
        if(biasTerm){
            biasBuffer = params[1];
        }
        else{
            biasBuffer = new byte[numOutput * 4];
        }
    }

    @Override
    public void setup(){
        initKernel();
    }

    private void initKernel() {
        int inputChannel = inputShape[0][3];
        int kernelHeight = kernelH;
        int kernelWidth = kernelW;
        int kernelSize = kernelHeight * kernelWidth;
        int inputChannelAligned = getInputChannelAligned();

        int totalKernelSize = inputChannelAligned * kernelSize;

        Allocation kernelAllocation;
        Allocation biasAllocation;

        Type.Builder kernelType = new Type.Builder(renderScript, Element.F32(renderScript));
        kernelType.setX(inputChannelAligned);
        kernelType.setY(kernelH * kernelW);
        Type biasType = Type.createX(renderScript, Element.F32(renderScript), inputChannelAligned);
        kernelAllocation = Allocation.createTyped(renderScript, kernelType.create(), Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation = Allocation.createTyped(renderScript, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        if(weightBuffer==null) {
            float[] kernelMatrix = new float[totalKernelSize];
            float[] biasArray = new float[inputChannelAligned];

            int count = 0;

            for (int j = 0; j < kernelHeight; j++) {
                for (int k = 0; k < kernelWidth; k++) {
                    for (int i = 0; i < inputChannelAligned; i++) {
                        if (i >= inputChannel) {
                            kernelMatrix[count++] = 0;
                        } else {
                            kernelMatrix[count++] = weight[i][0][j][k];
                        }
                    }
                }
            }

            for (int i = 0; i < inputChannelAligned; i++) {
                if (i >= inputChannel) {
                    biasArray[i] = 0;
                } else {
                    biasArray[i] = bias[i];
                }
            }

            kernelAllocation.copyFrom(kernelMatrix);
            biasAllocation.copyFrom(biasArray);
        }
        else {
            kernelAllocation.copyFromUnchecked(weightBuffer);
            biasAllocation.copyFromUnchecked(biasBuffer);
        }
        scriptDeconvolution = new ScriptC_Deconvolution(renderScript);
        scriptDeconvolution.set_BiasData(biasAllocation);
        scriptDeconvolution.set_KernelData(kernelAllocation);
        scriptDeconvolution.set_channelAligned(inputChannelAligned);
        scriptDeconvolution.set_padH(padH);
        scriptDeconvolution.set_padW(padW);
        scriptDeconvolution.set_strideH(strideH);
        scriptDeconvolution.set_strideW(strideW);
        scriptDeconvolution.set_kernelH(kernelH);
        scriptDeconvolution.set_kernelW(kernelW);

    }

    @Override
    public void computeFeatureMap() {
        FeatureMap input = (FeatureMap) featureMapInput[0];
        FeatureMap output = (FeatureMap) featureMapOutput;

        scriptDeconvolution.set_inputHeight(inputShape[0][1]);
        scriptDeconvolution.set_inputWidth(inputShape[0][2]);
        scriptDeconvolution.set_outputHeight(outputShape[1]);
        scriptDeconvolution.set_outputWidth(outputShape[2]);

        scriptDeconvolution.set_InputData(input.getFeatureMap());
        scriptDeconvolution.set_OutputData(output.getFeatureMap());
        Script.LaunchOptions options = new Script.LaunchOptions();
        options.setX(0, getOutputChannelAligned() / 4).setY(0, outputShape[1] * outputShape[2]);
        //TODO: support no grouped deconvolution
        scriptDeconvolution.forEach_deconv_dw4(options);
    }

    @Override
    public void computeOutputShape() {
        int h = inputShape[0][1];
        int w = inputShape[0][2];
        int c = inputShape[0][3];
        int n = inputShape[0][0];

        int outH = strideW * (h - 1) + kernelH - 2 * padH;
        int outW =  strideH * (w - 1) + kernelW - 2 * padW;

        outputShape = new int[]{n, outH, outW, c};
        allocFeatureMap();
    }
}

package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.Script;
import android.renderscript.Type;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.network.LayerParamInterface;
import com.rscnn.algorithm.ShapeUtils;
import com.rscnn.utils.LogUtil;

import layers.ScriptC_InnerProduct;

public class InnerProduct  extends Layer implements LayerParamInterface {
    private boolean biasTerm = true;  //whether or not has bias
    private int numOutput;

    private float[] weight;
    private float[] bias;

    private byte[] weightBuffer;
    private byte[] biasBuffer;

    private int inputAlign = 8;             //4 or 8 ,for renderscript batch

    private boolean nextRelu = false;

    private ScriptC_InnerProduct innerProductScript;

    private static final String TAG = "InnerProduct";

    public void setNumOutput(int numOutput) {
        this.numOutput = numOutput;
    }

    public void setBiasTerm(boolean biasTerm) {
        this.biasTerm = biasTerm;
    }

    public boolean isNextRelu() {
        return nextRelu;
    }

    public void setNextRelu(boolean nextRelu) {
        this.nextRelu = nextRelu;
    }

    private float[] reshapeTranspose(float[] input, int n, int c, int h, int w)
    {
        float[] output = new float[n * c * h * w];
        int count = 0;
        int chw = c * h * w;
        int cw = c * w;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<h;k++){
                    for(int l=0;l<w;l++){
                        output[i * chw + k * cw + l * c + j] = input[count++];
                    }
                }
            }
        }
        return output;
    }

    @Override
    public void setup()
    {
        int outputC = numOutput;
        int inputH = inputShape[0][1];
        int inputW = inputShape[0][2];
        int inputC = inputShape[0][3];

        if(weightBuffer==null) {
            if (inputH != 1 || inputW != 1) {//for weight reordering to nhwc
                weight = reshapeTranspose(weight, outputC, inputC, inputH, inputW);
            }
        }
        initKernel();
        this.inputAlign = 8;//seems that F8F1 is faster
    }

    @Override
    public void loadParams(byte[][] params) {
        weightBuffer = params[0];
        if(biasTerm){
            biasBuffer = params[1];
        }
        else{
            biasBuffer = new byte[getOutputChannelAligned() * 4];
        }
    }

    public InnerProduct() {
    }

    private void initKernel() {
        int n = outputShape[0];// in=out=200
        int c = outputShape[3];// 512

        int inh = inputShape[0][1];
        int inw = inputShape[0][2];
        int inc = inputShape[0][3];

        int inputChannelAlign = inc;
        if (inc % inputAlign != 0)
            inputChannelAlign = inc + inputAlign - inc % inputAlign;

        Type kernelType, biasType;
        Allocation kernelAllocation;
        Allocation biasAllocation;
        kernelType = Type.createX(renderScript, Element.F32_4(renderScript), c * inh * inw * inc / 4);
        biasType = Type.createX(renderScript, Element.F32(renderScript), c);

        kernelAllocation = Allocation.createTyped(renderScript, kernelType);
        biasAllocation = Allocation.createTyped(renderScript, biasType);

        if(weightBuffer==null) {
            kernelAllocation.copyFrom(weight);
            biasAllocation.copyFrom(bias);
        }
        else{
            kernelAllocation.copyFromUnchecked(weightBuffer);
            biasAllocation.copyFromUnchecked(biasBuffer);
        }
        innerProductScript = new ScriptC_InnerProduct(renderScript);
        innerProductScript.set_Bias_Blob(biasAllocation);
        innerProductScript.set_Kernel_Blob(kernelAllocation);
        innerProductScript.set_w_w(inc * inh * inw);
        innerProductScript.set_c_o(c);
        innerProductScript.set_c_i(inputChannelAlign * inh * inw);
        if(nextRelu){
            innerProductScript.set_relu(1);
        }
        else{
            innerProductScript.set_relu(0);
        }

        weight = null;
        bias = null;
    }

    @Override
    public void computeOutputShape()
    {
        int n = inputShape[0][0];
        outputShape = new int[]{n, 1, 1, numOutput};
        allocFeatureMapNoBlock();
        if(innerProductScript!=null){
            int inh = inputShape[0][1];
            int inw = inputShape[0][2];
            int inc = inputShape[0][3];
            int inputChannelAlign = inc;
            if (inc % inputAlign != 0)
                inputChannelAlign = inc + inputAlign - inc % inputAlign;

            innerProductScript.set_w_w(inc * inh * inw);
            innerProductScript.set_c_i(inputChannelAlign * inh * inw);
        }
    }

    @Override
    public void computeFeatureMap(){
        FeatureMap input = (FeatureMap) featureMapInput[0];
        Allocation frameAllocation = input.getFeatureMap();
        FeatureMap output = (FeatureMap) featureMapOutput;
        Allocation outAllocation = output.getFeatureMap();
        innerProductScript.set_In_Blob(frameAllocation);
        innerProductScript.set_Out_Blob(outAllocation);
        Script.LaunchOptions option = new Script.LaunchOptions();
        if(numOutput % 128==0){
            int thread_group = 1;
            option.setX(0, outputShape[0] * numOutput / 4 / thread_group);
            //option.setY(0, numOutput / 4 / thread_group);
            innerProductScript.set_thread_group(thread_group);
            innerProductScript.forEach_compute_f8fn_1(option);
        }
        else{
            option.setX(0, outputShape[0]);
            option.setY(0, numOutput);
            innerProductScript.forEach_compute_f8f1(option);
        }
    }
}

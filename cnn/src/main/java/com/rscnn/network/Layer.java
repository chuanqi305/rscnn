package com.rscnn.network;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Script;
import android.renderscript.ScriptIntrinsicBLAS;
import android.renderscript.Type;

public abstract class Layer {
    protected RenderScript renderScript;
    protected static ScriptIntrinsicBLAS scriptIntrinsicBLAS;
    protected boolean selfConnected = false;

    protected int[][] inputShape;
    protected int[] outputShape;
    protected Object[] featureMapInput;
    protected Object featureMapOutput;
    private boolean ready = false;
    protected String name;

    protected int getInputChannelAligned(){
        int ca = inputShape[0][3];
        if(ca % 4 != 0){
            ca = ca + 4 - ca % 4;
        }
        return ca;
    }

    protected int getOutputChannelAligned(){
        int ca = outputShape[3];
        if(ca % 4 != 0){
            ca = ca + 4 - ca % 4;
        }
        return ca;
    }

    protected static boolean weightFromTensorFlow = false;//tensorflow pooling implement is different from caffe
    protected static boolean reshapeFromTensorFlow = false;//tensorflow reshape use NHWC, caffe use NCHW

    public boolean isReady() {
        return ready;
    }

    public void setReady(boolean ready) {
        this.ready = ready;
    }

    public static void setWeightFromTensorFlow(boolean weightFromTensorFlow) {
        Layer.weightFromTensorFlow = weightFromTensorFlow;
    }

    public static void setReshapeFromTensorFlow(boolean reshapeFromTensorFlow) {
        Layer.reshapeFromTensorFlow = reshapeFromTensorFlow;
    }

    public void setName(String name) {
        this.name = name;
    }

    /**
     * layer is self-connected and can be processed on the input buffer, do not alloc new memory
     * @param selfConnected
     */
    public void setSelfConnected(boolean selfConnected) {
        this.selfConnected = selfConnected;
    }

    /**
     * if the layer is not last layer but has something output
     * @return
     */
    public boolean needOutput() {
        return false;
    }

    /**
     * do some initialize after all parameters are set
     */
    public void setup(){

    }

    public final void setRenderScript(RenderScript renderScript) {
        this.renderScript = renderScript;
        if(scriptIntrinsicBLAS==null) {
            scriptIntrinsicBLAS = ScriptIntrinsicBLAS.create(renderScript);
        }
    }

    protected void allocFeatureMap()
    {
        Type.Builder outputType = new Type.Builder(renderScript, Element.F32(renderScript));
        outputType.setZ(outputShape[0]);
        outputType.setY(outputShape[1] * outputShape[2]);
        outputType.setX(getOutputChannelAligned());
        Allocation outAllocation = Allocation.createTyped(renderScript, outputType.create());
        FeatureMap output = new FeatureMap();
        output.setFeatureMap(outAllocation);
        output.setN(outputShape[0]);
        output.setH(outputShape[1]);
        output.setW(outputShape[2]);
        output.setC(outputShape[3]);
        output.setPad4(true);
        if(this.featureMapOutput!=null){
            ((FeatureMap)featureMapOutput).getFeatureMap().destroy();
        }
        this.featureMapOutput = output;
    }

    protected void allocFeatureMapNoPad()
    {
        Type.Builder outputType = new Type.Builder(renderScript, Element.F32(renderScript));
        outputType.setZ(outputShape[0]);
        outputType.setY(outputShape[1] * outputShape[2]);
        outputType.setX(outputShape[3]);
        Allocation outAllocation = Allocation.createTyped(renderScript, outputType.create());
        FeatureMap output = new FeatureMap();
        output.setFeatureMap(outAllocation);
        output.setN(outputShape[0]);
        output.setH(outputShape[1]);
        output.setW(outputShape[2]);
        output.setC(outputShape[3]);
        output.setPad4(false);
        if(this.featureMapOutput!=null){
            ((FeatureMap)featureMapOutput).getFeatureMap().destroy();
        }
        this.featureMapOutput = output;
    }

    protected Script.LaunchOptions getLaunchOptionVector4(){
        Script.LaunchOptions options = new Script.LaunchOptions();
        options.setX(0, getOutputChannelAligned() / 4)
                .setY(0, outputShape[1] * outputShape[2]);
        return options;
    }

    protected void allocFeatureMapBlock4()
    {
        int outNum = outputShape[0];
        int outHeight = outputShape[1];
        int outWidth = outputShape[2];
        int outChannel = outputShape[3];

        int outChannelAlign = outChannel;
        if(outChannelAlign % 4 !=0) {
            outChannelAlign = outChannel + 4 - (outChannel % 4);
        }

        if(featureMapOutput!=null){
            FeatureMap old = (FeatureMap)featureMapOutput;
            if(old.getFeatureMap()!=null){
                Allocation out = old.getFeatureMap();
                if(out.getBytesSize()==outNum * outHeight * outWidth * outChannelAlign * 4){
                    old.setN(outNum);
                    old.setH(outHeight);
                    old.setW(outWidth);
                    old.setC(outChannel);
                    return;
                }
                else{
                    out.destroy();
                }
            }
        }

        Type outType = Type.createX(renderScript, Element.F32_4(renderScript), outNum * outHeight * outWidth * outChannelAlign / 4);
        //Allocation outAllocation = Allocation.createTyped(renderScript, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        Allocation outAllocation = Allocation.createTyped(renderScript, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_SCRIPT);
        FeatureMap output = new FeatureMap();
        output.setFeatureMap(outAllocation);
        output.setN(outNum);
        output.setH(outHeight);
        output.setW(outWidth);
        output.setC(outChannel);
        output.setPad4(true);
        output.setMatrix2D(false);
        featureMapOutput = output;

    }

    protected void allocFeatureMapNoBlock()
    {
        int outNum = outputShape[0];
        int outHeight = outputShape[1];
        int outWidth = outputShape[2];
        int outChannel = outputShape[3];

        if(featureMapOutput!=null){
            FeatureMap old = (FeatureMap)featureMapOutput;
            if(old.getFeatureMap()!=null){
                Allocation out = old.getFeatureMap();
                if(out.getBytesSize()==outNum * outHeight * outWidth * outChannel * 4){
                    old.setN(outNum);
                    old.setH(outHeight);
                    old.setW(outWidth);
                    old.setC(outChannel);
                    return;
                }
                else{
                    out.destroy();
                }
            }
        }

        Type outType = Type.createX(renderScript, Element.F32(renderScript), outNum * outHeight * outWidth * outChannel);
        //Allocation outAllocation = Allocation.createTyped(renderScript, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        Allocation outAllocation = Allocation.createTyped(renderScript, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_SCRIPT);
        FeatureMap output = new FeatureMap();
        output.setFeatureMap(outAllocation);
        output.setN(outNum);
        output.setH(outHeight);
        output.setW(outWidth);
        output.setC(outChannel);
        output.setMatrix2D(false);
        featureMapOutput = output;
    }

    public abstract void computeFeatureMap();

    public abstract void computeOutputShape();
}

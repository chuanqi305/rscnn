package com.rscnn.layers;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.algorithm.ShapeUtils;

import layers.ScriptC_ReshapeChannel2;

import static java.lang.System.arraycopy;

public class Reshape extends Layer {
    private int[] shapeDim;
    private int[] shapeDimParsed;
    private ScriptC_ReshapeChannel2 scriptReshape;

    public void setShapeDim(int[] shapeDim) {
        this.shapeDim = shapeDim;
    }

    public Reshape() {
    }

    @Override
    public void setup(){
        int channelAligned = getOutputChannelAligned();
        scriptReshape = new ScriptC_ReshapeChannel2(renderScript);
        scriptReshape.set_channelAligned(channelAligned);
    }

    @Override
    public void computeFeatureMap() {

        FeatureMap input = (FeatureMap) featureMapInput[0];
        FeatureMap output = (FeatureMap) featureMapOutput;

        float[][][][] in = input.getData();

        if(reshapeFromTensorFlow){
            float[][][][] out = ShapeUtils.reshape4(in, shapeDimParsed);
            output.setData(out);
            return;
        }

        float[][][][] in1 = FeatureMap.transeposeToNCHW(in);
        float[][][][] out = ShapeUtils.reshape4(in1, shapeDimParsed);
        float[][][][] out1 = FeatureMap.transeposeToNHWC(out);
        output.setData(out1);
    }

    @Override
    public void computeOutputShape() {
        int[] dim = new int[shapeDim.length];
        int n = inputShape[0][0];
        int c = inputShape[0][3];
        int w = inputShape[0][2];
        int h = inputShape[0][1];
        int old_dim[] = {n,c,h,w};

        if(reshapeFromTensorFlow){
            old_dim = new int[]{n, h, w, c};
        }

        arraycopy(shapeDim, 0, dim, 0, shapeDim.length);

        if(dim.length != 4){
            int[] dim2 = new int[4];
            for(int i=0;i < 4 - dim.length; i++){
                dim2[i] = inputShape[0][i];
            }
            for(int i=4-dim.length; i<4; i++){
                dim2[i] = dim[i - (4-dim.length)];
            }
            dim = dim2;
            //LogUtil.e("ReshapeLayer","parameter error:reshape dims must be 4.");
            //throw new IllegalArgumentException("reshape dims must be 4.");
        }

        for(int i=0;i<dim.length;i++)
            dim[i] = dim[i]!=0?dim[i]:old_dim[i];
        for(int i=0;i<dim.length;i++)
            dim[i] = dim[i]!=-1?dim[i]:-(n*c*w*h)/(dim[0]*dim[1]*dim[2]*dim[3]);

        shapeDimParsed = dim;
        if(reshapeFromTensorFlow){
            outputShape = dim.clone();
            allocFeatureMapNoBlock();//TODO: should use allocFeatureMapNoBlock
        }
        else{
            outputShape = new int[]{dim[0],dim[2],dim[3],dim[1]};
            int channels = outputShape[3];
            if(channels==2){//to channel2
                allocFeatureMapNoBlock();
            }
            else{
                allocFeatureMapBlock4();
            }
        }
    }
}

package com.rscnn.layers;

import android.renderscript.Allocation;
import android.renderscript.Script;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import layers.ScriptC_Concat;

/**
 * concat the data in channel axis.
 * eg:
 * input[0] = 1 * 16 * 300 * 400
 * input[1] = 1 * 24 * 300 * 400
 * output   = 1 * 40 * 300 * 400
 */
public class Concat extends Layer {
    private int axis;
    private ScriptC_Concat scriptConcat;

    public void setAxis(int axis) {
        this.axis = axis;
    }

    @Override
    public void setup()
    {
        int outChannelAligned = getOutputChannelAligned();

        int outBlockAligned = outChannelAligned / 4;

        scriptConcat = new ScriptC_Concat(renderScript);
        scriptConcat.set_outBlockAligned(outBlockAligned);
        scriptConcat.set_outChannelAligned(outChannelAligned);
        scriptConcat.set_height(outputShape[1]);
        scriptConcat.set_width(outputShape[2]);
    }

    @Override
    public void computeFeatureMap() {
        int blockOffset = 0;
        int channelOffset = 0;
        boolean useVector = true;
        FeatureMap output = (FeatureMap) featureMapOutput;
        Allocation outAllocation = output.getFeatureMap();

        int h = outputShape[1];
        int w = outputShape[2];

        scriptConcat.set_out_Blob(outAllocation);

        for(int i = 0; i< featureMapInput.length; i++){
            FeatureMap input = (FeatureMap) featureMapInput[i];
            Allocation frameAllocation = input.getFeatureMap();
            int inChannel = inputShape[i][3];
            int inChannelAligned = inChannel;
            if(inChannel % 4!=0){
                inChannelAligned = inChannel + 4 - inChannel % 4;
                useVector = false;
            }
            int inBlock = inChannelAligned / 4;// channels / 4
            scriptConcat.set_inBlockAligned(inBlock);
            scriptConcat.set_blockOffset(blockOffset);
            scriptConcat.set_inChannel(inChannel);
            scriptConcat.set_inChannelAligned(inChannelAligned);
            scriptConcat.set_channelOffset(channelOffset);
            scriptConcat.set_in_Blob(frameAllocation);
            Script.LaunchOptions option = new Script.LaunchOptions();
            if(useVector) {
                option.setX(0, inBlock).setY(0, h * w);
                scriptConcat.forEach_compute_in4out4(option);
            }
            else{
                option.setX(0, inChannel).setY(0, h * w);
                scriptConcat.forEach_compute(option);
            }
            blockOffset += inBlock;
            channelOffset += inChannel;
        }
    }

    @Override
    public void computeOutputShape() {
        int n = inputShape[0][0];
        int h = inputShape[0][1];
        int w = inputShape[0][2];
        int c = 0;
        for(int[] shape:inputShape){
            c += shape[3];
        }
        outputShape = new int[]{n, h, w, c};
        allocFeatureMap();
    }

}

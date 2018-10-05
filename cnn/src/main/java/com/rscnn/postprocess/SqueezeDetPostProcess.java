package com.rscnn.postprocess;

import android.graphics.Bitmap;

import com.rscnn.algorithm.NMS;
import com.rscnn.algorithm.ShapeUtils;
import com.rscnn.network.DetectResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SqueezeDetPostProcess extends PostProcess {

    private float[] outputBox;
    private float[] outputScore;
    private int[] outputClass;

    private int TOP_N_DETECTION = 64;
    private float PROB_THRESH = 0.005f;
    private float NMS_THRESH = 0.4f;
    private float PLOT_PROB_THRESH = 0.4f;

    private float[][] anchors;
    private int targetWidth = 1242;
    private int targetHeight = 375;

    private String[] classes = {"car", "pedestrian","cyclist"};
    private float[][] anchor = {
            {36.f,  37.f},
            {366.f, 174.f},
            {115.f,  59.f},
            {162.f,  87.f},
            {38.f,  90.f},
            {258.f, 173.f},
            {224.f, 108.f},
            {78.f, 170.f},
            {72.f,  43.f}
    };
    private int classCount = classes.length;

    int outH = 22;
    int outW = 76;
    int anchorCount = anchor.length;
    private int outputSize = outH * outW * anchorCount;


    public SqueezeDetPostProcess() {
        anchors = genAnchorBox();
        outputBox = new float[outputSize * 4];
        outputScore = new float[outputSize];
        outputClass = new int[outputSize];
    }

    private float[][] genAnchorBox()
    {
        int width = outW;
        int height = outH;
        int count = anchorCount;
        int outputCount = height * width * count;

        float[][][] centerX = new float[height][width][count];
        float[][][] centerY = new float[height][width][count];
        float[][] output = new float[outputCount][4];
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                for(int k=0;k<count;k++){
                    centerX[i][j][k] = (float)((j + 1) * targetWidth) / (float)(width + 1);
                }
            }
        }
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                for(int k=0;k<count;k++){
                    centerY[i][j][k] = (float)((i + 1) * targetHeight) / (float)(height + 1);
                }
            }
        }
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                for(int k=0;k<count;k++){
                    output[i * width * count + j * count + k][0] = centerX[i][j][k];
                    output[i * width * count + j * count + k][1] = centerY[i][j][k];
                    output[i * width * count + j * count + k][2] = anchor[k][0];
                    output[i * width * count + j * count + k][3] = anchor[k][1];
                }
            }
        }
        return output;
    }

    @Override
    public List<DetectResult> process(Bitmap image, NetworkParameter param, Map<String, Object> output) {
        float[][][][] outputValue = (float[][][][]) output.get((String)output.keySet().toArray()[0]);

        int count = 0;
        int channelPreds = classCount * anchorCount;
        int predConfChannelStart = channelPreds;
        int predConfChannelEnd = channelPreds + anchorCount;
        int deltaStart = predConfChannelEnd;
        int deltaEnd = deltaStart + anchorCount * 4;

        float[][][][] preds = new float[1][outH][outW][channelPreds];
        float[] pred_conf = new float[outputSize];
        float[][][][] pred_box_delta = new float[1][outH][outW][36];

        for(int i=0;i<outH;i++){
            for(int j=0;j<outW;j++){
                for(int k=0;k<channelPreds;k++){
                    preds[0][i][j][k] = outputValue[0][i][j][k];
                }
                for(int k=predConfChannelStart;k<predConfChannelEnd;k++){
                    pred_conf[count++] = outputValue[0][i][j][k];
                }
                for(int k=deltaStart;k<deltaEnd;k++){
                    pred_box_delta[0][i][j][k - deltaStart] = outputValue[0][i][j][k];
                }
            }
        }

        float[][] pred_class_probs = ShapeUtils.reshape4(preds, new int[]{1,1,outputSize,classCount})[0][0];
        float[][] pred_box_delta1 = ShapeUtils.reshape4(pred_box_delta, new int[]{1,1,outputSize,4})[0][0];
        float[] expSum = new float[outputSize];
        for(int i=0;i<pred_class_probs.length;i++){
            for(int j=0;j<pred_class_probs[0].length;j++){
                expSum[i] += Math.exp(pred_class_probs[i][j]);
            }
        }
        for(int i=0;i<outputSize;i++){
            for(int j=0;j<classCount;j++){
                pred_class_probs[i][j] = (float)Math.exp(pred_class_probs[i][j]) / expSum[i];
            }
        }

        //sigmoid
        for(int i=0;i<outputSize;i++){
            pred_conf[i] = 1.f / (1.f + (float) Math.exp(-pred_conf[i]));
        }

        for(int i=0;i<outputSize;i++){
            for(int j=0;j<classCount;j++){
                pred_class_probs[i][j] *=  pred_conf[i];
            }
        }

        for(int i=0;i<outputSize;i++){
            float anchor_x = anchors[i][0];
            float anchor_y = anchors[i][1];
            float anchor_w = anchors[i][2];
            float anchor_h = anchors[i][3];
            float delta_x = pred_box_delta1[i][0];
            float delta_y = pred_box_delta1[i][1];
            float delta_w = pred_box_delta1[i][2];
            float delta_h = pred_box_delta1[i][3];
            float box_center_x = anchor_x + delta_x * anchor_w;
            float box_center_y = anchor_y + delta_y * anchor_h;
            float box_width = anchor_w * (float)Math.exp(delta_w);
            float box_height = anchor_h * (float)Math.exp(delta_h);

            outputBox[i * 4] = box_center_x;
            outputBox[i * 4 + 1] = box_center_y;
            outputBox[i * 4 + 2] = box_width;
            outputBox[i * 4 + 3] = box_height;
        }

        for(int i=0;i<outputSize;i++){
            float max = 0;
            int index = 0;
            for(int j=0;j<3;j++){
                if(pred_class_probs[i][j]>max){
                    index = j;
                    max = pred_class_probs[i][j];
                }
            }
            outputScore[i] = max;
            outputClass[i] = index;
        }

        List<Integer> keep = new ArrayList<>();
        for(int i=0;i<outputSize;i++) {
            if (outputScore[i] > PROB_THRESH) {
                keep.add(i);
            }
        }

        int outSize = keep.size();

        float[][] outBox = new float[outSize][4];
        float[] outProb = new float[outSize];
        int[] outClass = new int[outSize];

        for(int i=0;i<keep.size();i++){
            int idx = keep.get(i);
            float x = outputBox[idx * 4];
            float y = outputBox[idx * 4 + 1];
            float w = outputBox[idx * 4 + 2];
            float h = outputBox[idx * 4 + 3];
            outBox[i][0] = x-w/2;
            outBox[i][1] = y-h/2;
            outBox[i][2] = x+w/2;
            outBox[i][3] = y+h/2;

            outBox[i][0] = Math.min(Math.max(outBox[i][0] ,0.f), targetWidth - 1);
            outBox[i][1] = Math.min(Math.max(outBox[i][1] ,0.f), targetHeight - 1);
            outBox[i][2] = Math.min(Math.max(outBox[i][2] ,0.f), targetWidth - 1);
            outBox[i][3] = Math.min(Math.max(outBox[i][3] ,0.f), targetHeight - 1);

            outProb[i] = outputScore[idx];
            outClass[i] = outputClass[idx];
        }

        int[] idxPerCls = new int[classCount];
        for(int i=0;i<classCount;i++){
            idxPerCls[i] = 0;
        }
        for(int i=0;i<keep.size();i++){
            if(outClass[i] >= classCount){
                continue;
            }
            idxPerCls[outClass[i]] ++;
        }

        List<DetectResult> result = new ArrayList<>();
        for(int i=0;i<classCount;i++){
            float[][] finalBox = new float[idxPerCls[i]][4];
            float[] finalProb = new float[idxPerCls[i]];
            count = 0;
            for(int j=0;j<outSize;j++){
                if(outClass[j]==i){
                    finalBox[count] = outBox[j];
                    finalProb[count] = outProb[j];
                    count++;
                }
            }
            NMS.sortScores(finalBox,finalProb);
            int[] index = NMS.nmsScoreFilter(finalBox, finalProb, TOP_N_DETECTION, NMS_THRESH);
            for(int j=0;j<index.length;j++){
                if(finalProb[index[j]] < PLOT_PROB_THRESH){
                    break;
                }
                float conf = finalProb[index[j]];
                String title = classes[i];
                DetectResult det = new DetectResult((int)finalBox[index[j]][0],(int)finalBox[index[j]][1], (int)finalBox[index[j]][2], (int)finalBox[index[j]][3],
                        title, conf);
                result.add(det);
            }
        }
        return result;
    }
}

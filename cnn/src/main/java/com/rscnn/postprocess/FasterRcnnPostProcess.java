package com.rscnn.postprocess;

import android.graphics.Bitmap;

import com.rscnn.algorithm.NMS;
import com.rscnn.algorithm.ShapeUtils;
import com.rscnn.network.DetectResult;
import com.rscnn.utils.LogUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class FasterRcnnPostProcess extends PostProcess {
    private final int MAX_BOX_PER_CLASS = 10;
    private final float MIN_CLASS_SCORE = 0.8f;

    private float NMS_THRESH = 0.3f;
    private String TAG = "FasterRcnnPostProcess";

    @Override
    public List<DetectResult> process(Bitmap image, NetworkParameter param,
                                      Map<String, Object> output) {
        int width = image.getWidth();
        int height = image.getHeight();
        int networkInputWidth = param.getNetworkInputWidth();
        int networkInputHeight = param.getNetworkInputHeight();
        List<DetectResult> result = new ArrayList<>();
        if(output.keySet().size()!=3){ //bbox_pred cls_pred bbox_delta
            LogUtil.e(TAG, "faster-rcnn network must have 3 outputs");
            return null;
        }

        Object[] out = new Object[output.keySet().size()];
        int count = 0;
        for(String key:output.keySet()){
            out[count++] = output.get(key);
        }
        float[][] box = getProposal(networkInputWidth, networkInputHeight, out, height, width);
        for(float[] b:box){
            int labelIndex = (int)b[4];
            String detection = labels[labelIndex];
            DetectResult res = new DetectResult((int)b[0], (int)b[1], (int)b[2], (int)b[3],
                    detection,b[5]);
            res.setIndex(labelIndex - 1);
            result.add(res);
        }

        return result;
    }

    private float[][][] transformInvByClass(float[][] boxes, float[][] deltas) {
        int boxCount = boxes.length;
        int classesCount = deltas[0].length / 4;
        float[][][] transBox = new float[classesCount][boxCount][4];//做成21 ＊ 300 ＊ 4 便于处理
        for(int i=0;i<boxCount;i++){
            for(int j=0;j<classesCount;j++){
                float x1 = boxes[i][0];
                float y1 = boxes[i][1];
                float x2 = boxes[i][2];
                float y2 = boxes[i][3];
                float dx = deltas[i][j*4];
                float dy = deltas[i][j*4+1];
                float dw = deltas[i][j*4+2];
                float dh = deltas[i][j*4+3];
                float height = y2 - y1 + 1.f;
                float width = x2 - x1 + 1.f;
                float ctrX = x1 + 0.5f * width;
                float ctrY = y1 + 0.5f * height;
                float predictCenterX = dx * width + ctrX;
                float predictCenterY = dy * height + ctrY;
                float predictWidth = (float) Math.exp(dw) * width;
                float predictHeight = (float) Math.exp(dh) * height;
                transBox[j][i][0] = predictCenterX - 0.5f * predictWidth;
                transBox[j][i][1] = predictCenterY - 0.5f * predictHeight;
                transBox[j][i][2] = predictCenterX + 0.5f * predictWidth;
                transBox[j][i][3] = predictCenterY + 0.5f * predictHeight;
            }
        }
        return transBox;
    }

    private static void clipBox(float[][][] box, int height, int width){
        for(float[][] b:box){
            clipBox(b,height,width);
        }
    }

    private static void clipBox(float[][] box, int height, int width){
        for(int i=0;i<box.length;i++){
            box[i][0] = Math.max(Math.min(box[i][0], (float)width - 1.f), 0.f);
            box[i][1] = Math.max(Math.min(box[i][1], (float)height - 1.f), 0.f);
            box[i][2] = Math.max(Math.min(box[i][2], (float)width - 1.f), 0.f);
            box[i][3] = Math.max(Math.min(box[i][3], (float)height - 1.f), 0.f);
        }
    }

    private float[][] getProposal(int networkInputWidth, int networkInputHeight,
                                  Object[] input, int origHeight, int origWidth){
        float[][] rawBoxes = (float[][]) input[0];//boxes, from proposal 300 * 4
        float[][] scores = (float[][]) input[1];//classes, 300 * 21
        float[][] deltas = (float[][])input[2];//deltas, 300 * 84

        float scaleX = (float) networkInputWidth / (float) origWidth;
        float scaleY = (float) networkInputHeight / (float) origHeight;

        int classCount = scores[0].length;//21
        for(int i=0;i<rawBoxes.length;i++){
            rawBoxes[i][0] /= scaleX;
            rawBoxes[i][1] /= scaleY;
            rawBoxes[i][2] /= scaleX;
            rawBoxes[i][3] /= scaleY;
        }

        float[][][] boxes = transformInvByClass(rawBoxes,deltas);//box for every delta, 21 * 300 * 4
        scores = ShapeUtils.transpose2D(scores);// scores to 21 * 300
        clipBox(boxes, origHeight, origWidth);
        List<float[]> boxAndScore = new ArrayList<>();
        for(int i=1;i<classCount;i++){//skip the background class
            NMS.sortScores(boxes[i],scores[i]);
            int[] index = NMS.nmsScoreFilter(boxes[i], scores[i], MAX_BOX_PER_CLASS, NMS_THRESH);
            if(index.length>0){
                for(int id:index){
                    if(scores[i][id] < MIN_CLASS_SCORE) break;
                    if(Float.isNaN(scores[i][id])){//skip the NaN score, maybe not correct
                        continue;
                    }
                    float[] boxScore = new float[6];
                    for (int j=0;j<4;j++)
                        boxScore[j] = boxes[i][id][j];//x1,y1,x2,y2
                    boxScore[4] = i;              //class index
                    boxScore[5] = scores[i][id];  //possibility
                    boxAndScore.add(boxScore);
                }
            }
        }
        float[][] out = new float[boxAndScore.size()][6];
        for(int i=0;i<out.length;i++){
            out[i] = boxAndScore.get(i);
        }
        return out;
    }
}

package com.rscnn.postprocess;

import android.graphics.Bitmap;

import com.rscnn.algorithm.NMS;
import com.rscnn.algorithm.ShapeUtils;
import com.rscnn.network.DetectResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RfcnPostProcess extends PostProcess {
    private int MAX_BOX_PER_CLASS = 10;
    private float NMS_THRESH = 0.4f;
    private float MIN_CLASS_SCORE = 0.3f;

    @Override
    public List<DetectResult> process(Bitmap image, NetworkParameter param, Map<String, Object> outputList) {
        List<DetectResult> result = new ArrayList<>();
        int width = image.getWidth();
        int height = image.getHeight();

        Object[] output = new Object[outputList.keySet().size()];
        int count = 0;
        for(String key: outputList.keySet()){
            output[count++] = outputList.get(key);
        }

        float[][] rois = (float[][])output[1];
        float[][] cls_prob = (float[][])output[2];
        float[][] bbox = (float[][])output[3];

        int roiSize = rois.length;
        float[][] bbox2 = new float[roiSize][4];
        for(int i=0;i<roiSize;i++){
            bbox2[i][0] = bbox[i][4];
            bbox2[i][1] = bbox[i][5];
            bbox2[i][2] = bbox[i][6];
            bbox2[i][3] = bbox[i][7];
        }

        float[][] finalBox = transformInvClip(bbox2, rois, height, width);
        int classCount = 6;

        float[][] scores = ShapeUtils.transpose2D(cls_prob);// scores to 21 * 300

        List<float[]> boxAndScore = new ArrayList<>();
        for(int i=1;i<classCount;i++){//skip the background class
            NMS.sortScores(finalBox,scores[i]);
            int[] index = NMS.nmsScoreFilter(finalBox, scores[i], MAX_BOX_PER_CLASS, NMS_THRESH);
            if(index.length>0){
                for(int id:index){
                    if(scores[i][id] < MIN_CLASS_SCORE) break;
                    if(Float.isNaN(scores[i][id])){//skip the NaN score, maybe not correct
                        continue;
                    }
                    float[] boxScore = new float[6];
                    for(int j=0;j<4;j++)
                        boxScore[j] = finalBox[id][j];//x1,y1,x2,y2
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
        for(float[] b:out){
            int labelIndex = (int)b[4];
            String detection = labels[labelIndex];
            DetectResult res = new DetectResult((int)b[0],(int)b[1],(int)b[2],(int)b[3],
                    detection,b[5]);
            result.add(res);
        }
        return result;
    }

    private float[][] transformInvClip(float[][] anchors, float[][] deltas, int imHeight, int imWidth){
        int n = anchors.length;//300
        float[][] transAnchors = new float[n][4];

        for(int i=0; i< n; i++) {
            float x1 = anchors[i][0];
            float y1 = anchors[i][1];
            float x2 = anchors[i][2];
            float y2 = anchors[i][3];
            float dx = deltas[i][0];
            float dy = deltas[i][1];
            float dw = deltas[i][2];
            float dh = deltas[i][3];
            float height = y2 - y1 + 1.f;
            float width = x2 - x1 + 1.f;
            float ctr_x = x1 + 0.5f * width;
            float ctr_y = y1 + 0.5f * height;
            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = (float) Math.exp(dw) * width;
            float pred_h = (float) Math.exp(dh) * height;
            transAnchors[i][0] = pred_ctr_x - 0.5f * pred_w;
            transAnchors[i][1] = pred_ctr_y - 0.5f * pred_h;
            transAnchors[i][2] = pred_ctr_x + 0.5f * pred_w;
            transAnchors[i][3] = pred_ctr_y + 0.5f * pred_h;
            transAnchors[i][0] = Math.max(Math.min(transAnchors[i][0], (float)imWidth - 1.f), 0.f);
            transAnchors[i][1] = Math.max(Math.min(transAnchors[i][1], (float)imHeight - 1.f), 0.f);
            transAnchors[i][2] = Math.max(Math.min(transAnchors[i][2], (float)imWidth - 1.f), 0.f);
            transAnchors[i][3] = Math.max(Math.min(transAnchors[i][3], (float)imHeight - 1.f), 0.f);
        }
        return transAnchors;
    }
}

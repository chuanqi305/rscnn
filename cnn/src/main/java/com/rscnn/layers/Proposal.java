package com.rscnn.layers;


import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;
import com.rscnn.algorithm.NMS;

public class Proposal extends Layer {

    private int featStride = 16;
    private int baseSize = 16;
    private int minSize = 16;
    private float[] ratio = new float[]{0.5f, 1.f, 2.f};
    private float[] scale = new float[]{8.f, 16.f, 32.f};
    private int preNmsTopn;
    private int postNmsTopn;
    private float nmsThresh;

    private float minScore = 0.001f;

    private float[][] anchors = null;
    private float[][][][] realAnchors;

    private final static float INVALID_ANCHOR = -10000.0f;

    public Proposal() {
    }

    public void setFeatStride(int featStride) {
        this.featStride = featStride;
    }

    public void setBaseSize(int baseSize) {
        this.baseSize = baseSize;
    }

    public void setMinSize(int minSize) {
        this.minSize = minSize;
    }

    public void setRatio(float[] ratio) {
        this.ratio = ratio;
    }

    public void setScale(float[] scale) {
        this.scale = scale;
    }

    public void setPreNmsTopn(int preNmsTopn) {
        this.preNmsTopn = preNmsTopn;
    }

    public void setPostNmsTopn(int postNmsTopn) {
        this.postNmsTopn = postNmsTopn;
    }

    public void setNmsThresh(float nmsThresh) {
        this.nmsThresh = nmsThresh;
    }

    @Override
    public void setup(){

    }

    private float[][] generateAnchors()
    {
        int numRatios = ratio.length;
        int numScales = scale.length;
        float base_area = baseSize * baseSize;
        float center = 0.5f * ((float) baseSize - 1.f);
        float[][] anchors = new float[numRatios * numScales][4];
        for (int i = 0; i < numRatios; ++i) {
            float ratio_w = Math.round(Math.sqrt(base_area / ratio[i]));
            float ratio_h = Math.round(ratio_w * ratio[i]);
            for (int j = 0; j < numScales; ++j) {
                float scale_w = 0.5f * (ratio_w * scale[j] - 1.f);
                float scale_h = 0.5f * (ratio_h * scale[j] - 1.f);
                int index = i*numScales + j;
                anchors[index][0] = center - scale_w;
                anchors[index][1] = center - scale_h;
                anchors[index][2] = center + scale_w;
                anchors[index][3] = center + scale_h;
            }
        }
        return anchors;
    }


    private float[][][][] generateRealAnchors(int height, int width, int stride){
        float[][]anchors = this.anchors;
        int length = anchors.length;
        float[][][][] realAnchors = new float[length][height][width][4];
        for(int i=0;i<length;i++){
            for(int j=0;j<height;j++) {
                for (int k = 0; k < width; k++) {
                    realAnchors[i][j][k][0] = anchors[i][0] + k * stride;
                    realAnchors[i][j][k][1] = anchors[i][1] + j * stride;
                    realAnchors[i][j][k][2] = anchors[i][2] + k * stride;
                    realAnchors[i][j][k][3] = anchors[i][3] + j * stride;
                }
            }
        }
        return realAnchors;
    }

    private float[][] transformInv(float[][][][] anchors, float[] deltas){
        int n = anchors.length;
        int h = anchors[0].length;
        int w = anchors[0][0].length;
        int n_anchors = n * h *w;
        float[][] transAnchors = new float[n_anchors][4];
        int count=0;
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                for(int i=0; i< n; i++) {
                    float x1 = anchors[i][j][k][0];
                    float y1 = anchors[i][j][k][1];
                    float x2 = anchors[i][j][k][2];
                    float y2 = anchors[i][j][k][3];
                    float dx = deltas[count * 4];
                    float dy = deltas[count * 4 + 1];
                    float dw = deltas[count * 4 + 2];
                    float dh = deltas[count * 4 + 3];
                    float height = y2 - y1 + 1.f;
                    float width = x2 - x1 + 1.f;
                    float ctr_x = x1 + 0.5f * width;
                    float ctr_y = y1 + 0.5f * height;
                    float pred_ctr_x = dx * width + ctr_x;
                    float pred_ctr_y = dy * height + ctr_y;
                    float pred_w = (float) Math.exp(dw) * width;
                    float pred_h = (float) Math.exp(dh) * height;
                    transAnchors[count][0] = pred_ctr_x - 0.5f * pred_w;
                    transAnchors[count][1] = pred_ctr_y - 0.5f * pred_h;
                    transAnchors[count][2] = pred_ctr_x + 0.5f * pred_w;
                    transAnchors[count][3] = pred_ctr_y + 0.5f * pred_h;
                    count++;
                }
            }
        }
        return transAnchors;
    }

    private float[] reshapeScore(float[][][][] scores){
        int h = scores[0].length;
        int w = scores[0][0].length;
        int c = scores[0][0][0].length;
        int halfc = c / 2;

        float[] new_score = new float[halfc * w * h];

        int count = 0;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for(int k=halfc; k< c; k++) {
                    new_score[count++] = scores[0][i][j][k];
                }
            }
        }
        return new_score;
    }

    private static void clipBox(float[][] box, int height, int width){
        for(int i=0;i<box.length;i++){
            box[i][0] = Math.max(Math.min(box[i][0], (float)width - 1.f), 0.f);
            box[i][1] = Math.max(Math.min(box[i][1], (float)height - 1.f), 0.f);
            box[i][2] = Math.max(Math.min(box[i][2], (float)width - 1.f), 0.f);
            box[i][3] = Math.max(Math.min(box[i][3], (float)height - 1.f), 0.f);
        }
    }

    private static Object[] filterBox(float[][] anchors, float[] scores, float minSizeH, float minSizeW, float minScore){
        int count = anchors.length;

        for(int i=0;i<anchors.length;i++){
            if(anchors[i][2]-anchors[i][0]<minSizeW||anchors[i][3]-anchors[i][1]<minSizeH) {
                scores[i] = INVALID_ANCHOR;
                count--;
                continue;
            }
            if(scores[i] < minScore){
                scores[i] = INVALID_ANCHOR;
                count--;
            }
        }

        float[][] newBox = new float[count][4];
        float[] newScore = new float[count];

        count = 0;
        for(int i=0;i<anchors.length;i++){
            if(scores[i]!=INVALID_ANCHOR){
                newBox[count] = anchors[i];
                newScore[count] = scores[i];
                count++;
            }
        }

        return new Object[]{newBox, newScore};
    }

    private Object[] getAnchorsTopN(float[][] anchors, float[] scores, int n){
        Object out[] = new Object[2];
        int anchorCount = anchors.length;

        if(n >= anchorCount){
            out[0] = anchors;
            out[1] = scores;
            return out;
        }
        float[][]topA = new float[n][4];
        float[]topS = new float[n];
        int count = 0;
        for(int i=0;i<anchorCount;i++){
            if(scores[i] != INVALID_ANCHOR){
                topA[count] = anchors[i];
                topS[count++] = scores[i];
                if(count>=n)
                    break;
            }
        }
        out[0] = topA;
        out[1] = topS;
        return out;
    }

    private float[][] invoke(float[][][][]scores, float[] boxDeltas, float[] imInfo){

        int imHeight = (int)imInfo[0];
        int imWidth = (int)imInfo[1];

        float imScaleW = imInfo[2];
        float imScaleH = imInfo[3];

        float[][][][] anchors = realAnchors;

        float[] score = reshapeScore(scores);


        float[][] proposal = transformInv(anchors, boxDeltas);

        clipBox(proposal,imHeight,imWidth);

        Object[] filteredBox = filterBox(proposal, score, (float)minSize * imScaleH, (float)minSize * imScaleW, minScore);
        proposal = (float[][]) filteredBox[0];
        score = (float[])filteredBox[1];

        NMS.sortScores(proposal,score);

        Object proposalScore[] = getAnchorsTopN(proposal, score, preNmsTopn);

        int[] outputIndex = NMS.nmsScoreFilter((float[][]) proposalScore[0], (float[]) proposalScore[1], postNmsTopn, nmsThresh);

        proposal = new float[postNmsTopn][4];
        float[][] p1 = (float[][]) proposalScore[0];
        for(int i=0;i<outputIndex.length;i++){
            proposal[i] = p1[outputIndex[i]];
        }
        return proposal;
    }

    @Override
    public void computeFeatureMap() {

        FeatureMap scores = (FeatureMap) featureMapInput[0];
        FeatureMap boxDeltas = (FeatureMap) featureMapInput[1];
        float[] imInfo = (float[]) featureMapInput[2];
        this.preNmsTopn = (int)imInfo[4];
        this.postNmsTopn = (int)imInfo[5];
        this.nmsThresh = imInfo[6];

        FeatureMap outputData = (FeatureMap) featureMapOutput;

        float[][][][] score1 = scores.getData();
        float[] boxDelta1 = boxDeltas.getData1D();
        float[][] output = invoke(score1, boxDelta1, imInfo);
        outputData.setData(output);
    }

    @Override
    public void computeOutputShape() {
        outputShape = new int[]{1, 1, postNmsTopn, 4};
        allocFeatureMapNoBlock();
        this.anchors = generateAnchors();
        this.realAnchors = generateRealAnchors(inputShape[0][1], inputShape[0][2], featStride);
    }

    @Override
    public boolean needOutput()
    {
        return true;
    }
}

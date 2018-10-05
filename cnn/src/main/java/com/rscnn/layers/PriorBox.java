package com.rscnn.layers;

import com.rscnn.network.FeatureMap;
import com.rscnn.network.Layer;

import java.util.LinkedList;
import java.util.List;

public class PriorBox extends Layer {
    private float[] minSize;
    private float[] maxSize;
    private float[] aspectRatio;
    private boolean flip = true;
    private boolean clip = false;
    private float[] variance;
    private int imgH;
    private int imgW;
    private float stepH;
    private float stepW;
    private float offset = 0.5f;

    private float[][] output = null;

    private int numPriors;

    public void setMinSize(float[] minSize) {
        this.minSize = minSize;
    }

    public void setMaxSize(float[] maxSize) {
        this.maxSize = maxSize;
    }

    public void setAspectRatio(float[] aspectRatio) {
        this.aspectRatio = aspectRatio;
    }

    public void setVariance(float[] variance) {
        this.variance = variance;
    }

    public void setFlip(boolean flip) {
        this.flip = flip;
    }

    public void setClip(boolean clip) {
        this.clip = clip;
    }

    public void setImgSize(int imgSize) {
        this.imgH = this.imgW = imgSize;
    }

    public void setImgH(int imgH) {
        this.imgH = imgH;
    }

    public void setImgW(int imgW) {
        this.imgW = imgW;
    }

    public void setStep(float step) {
        this.stepH = this.stepW = step;
    }

    public void setStepH(float stepH) {
        this.stepH = stepH;
    }

    public void setStepW(float stepW) {
        this.stepW = stepW;
    }

    public void setOffset(float offset) {
        this.offset = offset;
    }

    @Override
    public void computeFeatureMap() {
        int height = inputShape[0][1];
        int width = inputShape[0][2];
        int imgHeight = inputShape[1][1];
        int imgWidth = inputShape[1][2];
        if(imgH != 0 && imgW != 0){
            imgHeight = imgH;
            imgWidth = imgW;
        }
        float stepH = (float)imgHeight / height;
        float stepW = (float)imgWidth / width;
        if(this.stepH != 0 && this.stepW != 0){
            stepH = this.stepH;
            stepW = this.stepW;
        }
        int dim = outputShape[3];
        if(output!=null){//compute once then use cache
            ((FeatureMap)featureMapOutput).setData(new float[][][][]{{output}});
            return;
        }
        output = new float[2][dim];

        int idx = 0;
        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                float centerX = (w + offset) * stepW;
                float centerY = (h + offset) * stepH;
                for(int s=0; s < minSize.length; s++){
                    float minsize = minSize[s];
                    float boxWidth = minsize;
                    float boxHeight = minsize;
                    // xmin
                    output[0][idx++] = (centerX - boxWidth / 2.f) / imgWidth;
                    // ymin
                    output[0][idx++] = (centerY - boxHeight / 2.f) / imgHeight;
                    // xmax
                    output[0][idx++] = (centerX + boxWidth / 2.f) / imgWidth;
                    // ymax
                    output[0][idx++] = (centerY + boxHeight / 2.f) / imgHeight;

                    if (maxSize.length > 0) {
                        float maxsize = maxSize[s];
                        boxWidth = (float)Math.sqrt(minsize * maxsize);
                        boxHeight = (float)Math.sqrt(minsize * maxsize);
                        output[0][idx++] = (centerX - boxWidth / 2.f) / imgWidth;
                        output[0][idx++] = (centerY - boxHeight / 2.f) / imgHeight;
                        output[0][idx++] = (centerX + boxWidth / 2.f) / imgWidth;
                        output[0][idx++] = (centerY + boxHeight / 2.f) / imgHeight;
                    }

                    for (int r = 0; r < aspectRatio.length; r++) {
                        float ar = aspectRatio[r];
                        if (Math.abs(ar - 1.) < 1e-6) {
                            continue;
                        }
                        boxWidth = minsize * (float)Math.sqrt(ar);
                        boxHeight = minsize / (float)Math.sqrt(ar);
                        output[0][idx++] = (centerX - boxWidth / 2.f) / imgWidth;
                        output[0][idx++] = (centerY - boxHeight / 2.f) / imgHeight;
                        output[0][idx++] = (centerX + boxWidth / 2.f) / imgWidth;
                        output[0][idx++] = (centerY + boxHeight / 2.f) / imgHeight;
                    }
                }
            }
        }
        if(clip){
            for(int i=0;i<dim;i++){
                output[0][i] = Math.min(Math.max(output[0][i], 0.f), 1.f);
            }
        }
        if(variance.length==1){
            for(int i=0; i<dim; i++){
                output[1][i] = variance[0];
            }
        }
        else{
            int points = dim / 4;
            for(int i=0; i<points; i++){
                output[1][i * 4] = variance[0];
                output[1][i * 4 + 1] = variance[1];
                output[1][i * 4 + 2] = variance[2];
                output[1][i * 4 + 3] = variance[3];
            }
        }
        ((FeatureMap)featureMapOutput).setData(new float[][][][]{{output}});
    }

    @Override
    public void computeOutputShape() {
        int layerHeight = inputShape[0][1];
        int layerWidth = inputShape[0][2];
        if(maxSize==null){
            maxSize = new float[0];
        }

        List<Float> aspectRatio1 = new LinkedList<>();
        aspectRatio1.add(1.f);
        for(int i=0; i < aspectRatio.length; i++){
            float ar = aspectRatio[i];
            boolean alreadyExist = false;
            for(int j=0;j<aspectRatio1.size();j++){
                if(ar == aspectRatio1.get(j)){
                    alreadyExist = true;
                    break;
                }
            }
            if(!alreadyExist){
                aspectRatio1.add(ar);
                if(flip){
                    aspectRatio1.add(1.f / ar);
                }
            }
        }

        aspectRatio = new float[aspectRatio1.size()];

        for(int i=0; i< aspectRatio1.size(); i++){
            aspectRatio[i] = aspectRatio1.get(i);
        }

        if(variance==null){
            variance = new float[]{0.1f};
        }

        numPriors = aspectRatio.length * minSize.length;
        if(maxSize.length > 0){
            numPriors += maxSize.length;// maxSize == minSize
        }
        outputShape = new int[]{inputShape[0][0], 1, 2, layerHeight * layerWidth * numPriors * 4};
        allocFeatureMapNoPad();
    }
}

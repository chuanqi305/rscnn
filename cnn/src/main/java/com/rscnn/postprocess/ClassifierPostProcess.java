package com.rscnn.postprocess;

import android.graphics.Bitmap;

import com.rscnn.algorithm.ArgSort;
import com.rscnn.network.DetectResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ClassifierPostProcess extends PostProcess {

    private int topN = 1;

    @Override
    public List<DetectResult> process(Bitmap image, NetworkParameter param, Map<String, Object> output) {
        List<DetectResult> result = new ArrayList<>();
        float[][] out = (float[][])output.get((String)output.keySet().toArray()[0]);
        float[] cls = out[0];

        if(topN == 1) {
            int labelIndex = 0;
            for (int i = 0; i < cls.length; i++) {
                if (cls[i] > cls[labelIndex]) {
                    labelIndex = i;
                }
            }
            String detection = labels[labelIndex];
            DetectResult res = new DetectResult(detection, cls[labelIndex]);
            res.setIndex(labelIndex);
            result.add(res);
        }
        else {
            int index[] = ArgSort.sort(cls);
            for (int i = 0; i < topN; i++) {
                int labelIndex = index[i];
                String detection = labels[labelIndex];
                DetectResult res = new DetectResult(detection, cls[labelIndex]);
                res.setIndex(labelIndex);
                result.add(res);
            }
        }
        return result;
    }
}

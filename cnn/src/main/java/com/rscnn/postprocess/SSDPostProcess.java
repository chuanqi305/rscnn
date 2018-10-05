package com.rscnn.postprocess;

import android.graphics.Bitmap;
import com.rscnn.network.DetectResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SSDPostProcess extends PostProcess {
    @Override
    public List<DetectResult> process(Bitmap image, NetworkParameter param, Map<String, Object> output) {
        float[][] box = (float[][]) output.get("detection_out");
        List<DetectResult> result = new ArrayList<>();
        if(box==null || box.length == 0 || box[0][0] < 0){
            return result;
        }
        int w = image.getWidth();
        int h = image.getHeight();
        for(float[] b:box){
            int labelIndex = (int)b[4];
            String detection = labels[labelIndex];
            DetectResult res = new DetectResult((int)(b[0] * w),
                    (int)(b[1] * h), (int)(b[2] * w), (int)(b[3] * h),
                    detection, b[5]);
            res.setIndex(labelIndex - 1);
            result.add(res);
        }
        return result;
    }
}

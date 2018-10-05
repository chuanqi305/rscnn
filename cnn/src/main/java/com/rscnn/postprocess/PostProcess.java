package com.rscnn.postprocess;

import android.graphics.Bitmap;

import com.rscnn.network.DetectResult;

import java.util.List;
import java.util.Map;

public abstract class PostProcess {
    protected String[] labels = new String[]{"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

    public abstract List<DetectResult> process(Bitmap image, NetworkParameter param, Map<String, Object> output);

    public void setLabels(String[] labels) {
        this.labels = labels;
    }
}

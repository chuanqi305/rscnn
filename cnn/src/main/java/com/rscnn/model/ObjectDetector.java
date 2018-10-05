package com.rscnn.model;

import android.graphics.Bitmap;

import com.rscnn.network.ConvNet;
import com.rscnn.network.DetectResult;

import java.util.List;

public abstract class ObjectDetector {

    protected ConvNet convNet;

    public List<DetectResult> detect(Bitmap image) {
        return convNet.detect(image);
    }
}

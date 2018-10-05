package com.rscnn.utils;

import android.content.res.AssetManager;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class LabelUtils {
    private static String TAG = "LabelUtils";
    public static String[] loadLabels(AssetManager assetManager, String file){
        try {
            String[] labels;
            InputStream labelFile = assetManager.open(file);
            byte[] buffer = new byte[labelFile.available()];
            labelFile.read(buffer);
            labelFile.close();
            String labelString = new String(buffer);
            labels = labelString.split("\n");
            for(int i=0;i<labels.length;i++){
                labels[i] = labels[i].split(",")[0].trim();
            }
            return labels;
        } catch (IOException e) {
            LogUtil.e(TAG, "load label error:" + e.getMessage());
            LogUtil.e(TAG, "use default voc 20 class label");
            e.printStackTrace();
        }
        return null;
    }

    private static String[] loadLabels(String file){
        try {
            String[] labels;
            InputStream labelFile = new FileInputStream(file);
            byte[] buffer = new byte[labelFile.available()];
            labelFile.read(buffer);
            labelFile.close();
            String labelString = new String(buffer);
            labels = labelString.split("\n");
            for(int i=0;i<labels.length;i++){
                labels[i] = labels[i].split(",")[0].trim();
            }
            return labels;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}

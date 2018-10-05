package com.rscnn.network;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.renderscript.RenderScript;

import com.rscnn.postprocess.NetworkParameter;
import com.rscnn.postprocess.PostProcess;
import com.rscnn.preprocess.PreProcess;
import com.rscnn.utils.LogUtil;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class ConvNet {
    private LayerGraph layer;

    private final static String TAG = "ConvNet";

    private int networkInputHeight;
    private int networkInputWidth;
    private PreProcess preProcess;
    private PostProcess postProcess;

    public ConvNet(RenderScript renderScript, AssetManager assetManager, String modelDir,
                   PreProcess preProcess, PostProcess postProcess) throws IOException {
        this.preProcess = preProcess;
        this.postProcess = postProcess;

        if(modelDir.endsWith("/")){
            modelDir = modelDir.substring(0, modelDir.length() - 1);
        }
        String[] fileList;
        if(assetManager==null) {
            File dir = new File(modelDir);
            File[] files = dir.listFiles();

            if (!dir.exists() || files == null || files.length == 0) {
                String errorMsg = "model dir " + modelDir + " is empty or permission denied.";
                LogUtil.e(TAG, errorMsg);
                throw new IOException(errorMsg);
            }
            fileList = new String[files.length];
            for(int i=0;i<files.length;i++){
                fileList[i] = files[i].getName();
            }
        }
        else{
            try {
                fileList = assetManager.list(modelDir);
            } catch (IOException e) {
                String errorMsg = "model dir assets://" + modelDir + " read error:" + e.getMessage();
                LogUtil.e(TAG, errorMsg);
                throw e;
            }
        }

        String prototxt = null;
        for(String f:fileList){
            if(f.endsWith(".prototxt") || f.endsWith(".proto")){
                prototxt = modelDir + "/" + f;
            }
            if(prototxt != null){
                break;
            }
        }

        if(prototxt==null){
            String errorMsg = "model dir " + modelDir + " must contains a .prototxt file";
            LogUtil.e(TAG, errorMsg);
            throw new IOException(errorMsg);
        }

        loadGraphFromRawData(renderScript, assetManager, prototxt, modelDir);
    }

    private void loadGraphFromRawData(RenderScript renderScript, AssetManager assetManager, String prototxt, String dataDir) throws IOException {
        LayerParser parser = new LayerParser(renderScript);
        try {
            if(assetManager!=null) {
                layer = parser.parseFromRawDataOnAsset(prototxt, assetManager, dataDir);
            }
            else{
                layer = parser.parseFromRawDataOnStorage(prototxt, dataDir);
            }
            if(layer==null){
                throw new IOException("load model error");
            }

            int[] inputDims = layer.getInputSize();
            this.networkInputHeight = inputDims[2];
            this.networkInputWidth = inputDims[3];
            layer.init();

        } catch (IOException e) {
            LogUtil.e(TAG, "load model error");
            e.printStackTrace();
            throw e;
        }
    }

    public int getNetworkInputHeight() {
        return networkInputHeight;
    }

    public int getNetworkInputWidth() {
        return networkInputWidth;
    }

    public void resetNetworkInputSize(int width, int height){
        this.networkInputHeight = height;
        this.networkInputWidth = width;
        layer.setInputSize(networkInputWidth, networkInputHeight);
    }

    public List<DetectResult> detect(Bitmap bmp){

        NetworkParameter param = new NetworkParameter();
        param.setNetworkInputHeight(networkInputHeight);
        param.setNetworkInputWidth(networkInputWidth);

        long temp = System.currentTimeMillis();
        Object[] input = preProcess.process(bmp, param);
        layer.setInputData(input);
        Map<String, Object> output =  layer.execute();
        List<DetectResult> result= postProcess.process(bmp, param, output);
        temp = System.currentTimeMillis() - temp;
        LogUtil.i("ConvNet","total: " + temp + " ms.");
        return result;
    }

    public int[][] segmentation(Bitmap bmp) {
        int height = bmp.getHeight();
        int width = bmp.getWidth();

        NetworkParameter param = new NetworkParameter();
        param.setNetworkInputHeight(networkInputHeight);
        param.setNetworkInputWidth(networkInputWidth);

        Object[] input = preProcess.process(bmp, param);
        layer.setInputData(input);
        Map<String, Object> output =  layer.execute();

        float[][][][] mask = ((float[][][][])output.get((String)output.keySet().toArray()[0]));
        int[][] newMask = new int[height][width];
        int outH = mask[0].length;
        int outW = mask[0][0].length;
        float factorH = (float) outH / (float)height;
        float factorW = (float) outW / (float)width;
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                int indexH = (int)Math.min(i * factorH, outH);
                int indexW = (int)Math.min(j * factorW, outW);
                newMask[i][j] =(int) mask[0][indexH][indexW][0];
            }
        }
        return newMask;
    }
}

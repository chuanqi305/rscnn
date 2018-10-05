package com.rscnn.network;

import com.rscnn.layers.Convolution;
import com.rscnn.layers.InnerProduct;
import com.rscnn.layers.Input;
import com.rscnn.layers.ReLU;
import com.rscnn.utils.LogUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LayerGraph {
    private String name;
    private String type;
    private List<LayerGraph> nextLayer;
    private Layer currentLayer;

    private String outPutName;
    private List<String> nextLayerNames;
    private List<String> prevLayerNames;

    private Object[] inputData;// TODO: not thread safe

    public LayerGraph() {
        nextLayer = new ArrayList<>();
        nextLayerNames = new ArrayList<>();
        prevLayerNames = new ArrayList<>();
    }

    public void initInputData()
    {
        int len = prevLayerNames.size();
        inputData = new Object[len];
        currentLayer.featureMapInput = new Object[len];
        currentLayer.inputShape = new int[len][0];
    }

    public void setInputData(Object[] inputData) {
        this.inputData = inputData;
    }

    public void setCurrentLayer(Layer currentLayer) {
        this.currentLayer = currentLayer;
    }

    public void setOutPutName(String outPutName) {
        this.outPutName = outPutName;
    }

    public Layer getCurrentLayer() {
        return currentLayer;
    }

    public List<LayerGraph> getNextLayer() {
        return nextLayer;
    }

    public List<String> getNextLayerNames() {
        return nextLayerNames;
    }

    public List<String> getPrevLayerNames() {
        return prevLayerNames;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    private boolean isAllInputDataReady(LayerGraph nextLayer){
        for(Object in:nextLayer.currentLayer.featureMapInput){
            if(in==null){
                return false;
            }
        }
        return true;
    }

    private boolean isAllInputShapeReady(LayerGraph nextLayer){
        for(int[] inputShape:nextLayer.currentLayer.inputShape){
            if(inputShape.length==0){
                return false;
            }
        }
        return true;
    }

    public int getNextInputIndex(LayerGraph nextLayer){
        List<String>inputs = nextLayer.getPrevLayerNames();
        for(int i=0;i<inputs.size();i++){
            if(inputs.get(i).equals(outPutName)){
                return  i;
            }
        }
        return 0;
    }

    private void debugShape(Object input,String name)
    {
        if(input.getClass().equals(float[][][][].class)){
            float[][][][] data = (float[][][][]) input;
            int n = data.length;
            int c = data[0].length;
            int h = data[0][0].length;
            int w = data[0][0][0].length;
            LogUtil.i("LayerGraph",name + " shape:"+n+","+c+","+h+","+w);
        }
        else if(input.getClass().equals(float[][].class)){
            float[][] data = (float[][]) input;
            int h = data.length;
            int w = data[0].length;
            LogUtil.i("LayerGraph",name + " shape:"+h+","+w);
        }
    }

    private void debugDatashape(Object[] input, Object output, String name){
        for(Object i:input) {
            debugShape(i, name+" input ");
        }
        debugShape(output,name+" output ");
    }

    private void execute(List<String> outputName, List<Object> outputList){
        long temp = System.currentTimeMillis();
//        LogUtil.i("LayerGraph","layer "+name+" compute start");

        currentLayer.computeFeatureMap();//run this layer

        Object output = currentLayer.featureMapOutput;
//        float[][][][] outputData = new float[0][0][0][0];
//
//        if(output instanceof FeatureMap) {
//            outputData = ((FeatureMap) output).getData();
//        }
//        int len = outputData.length;
        temp = System.currentTimeMillis() - temp;
        LogUtil.i("LayerGraph","compute time: "+name+" "+type+" " + temp + " ms.");

        for(int i = 0; i<currentLayer.featureMapInput.length; i++){
            currentLayer.featureMapInput[i] = null;
        }
        if(currentLayer.needOutput() || this.nextLayer.isEmpty()){//last layer output
            //LogUtil.i("LayerGraph", "get output["+outputList.size()+"] at layer "+name);
            outputName.add(currentLayer.name);
            outputList.add(output);
        }
        if(this.nextLayer.size()==0){
            return;
        }

        for(LayerGraph next:this.getNextLayer()) {
            if(currentLayer instanceof Input){//input 层有多个输出，分别输出到不同的节点
                Object [] out = (Object[])output;
                next.currentLayer.featureMapInput = new Object[out.length];
                System.arraycopy(out,0,next.currentLayer.featureMapInput,0,out.length);
            }
            else {
                int index = getNextInputIndex(next);
                next.currentLayer.featureMapInput[index] = output;
            }
        }

        for(LayerGraph next:this.getNextLayer()) {
            if(isAllInputDataReady(next)){//数据准备好后，计算下一个节点
                next.execute(outputName, outputList);
            }
        }
    }

    public Map<String, Object> execute(){
        List<Object> list = new ArrayList<>();
        List<String> nameList = new ArrayList<>();
        long temp = System.currentTimeMillis();

        currentLayer.featureMapInput = inputData;
        execute(nameList, list);
        LogUtil.w("LayerGraph","exec time: total time " + (System.currentTimeMillis() - temp) + " ms.");
        HashMap<String, Object> output = new LinkedHashMap<>();
        for(int i=0;i<list.size();i++){
            Object out = list.get(i);
            String name = nameList.get(i);
            if(out instanceof FeatureMap){
                LogUtil.d("LayerGraph", "output " + i+":"+out);
                FeatureMap ou = (FeatureMap)out;
                if(ou.getH()==1 && ou.getW()==1) {
                    output.put(name, ou.getData2D());
                }
                else{
                    output.put(name, ou.getData());
                }
            }
            else {
                output.put(name, out);
            }
        }

        temp = System.currentTimeMillis() - temp;
        LogUtil.w("LayerGraph","compute time: total time " + temp + " ms.");
        return output;
    }

    public int[] getInputSize() {
        Input input = (Input) currentLayer;
        return input.getDim();
    }

    public void setInputSize(int width, int height){
        Input input = (Input) currentLayer;
        input.setInputHeight(height);
        input.setInputWidth(width);
        clearShape(this);
        resizeInput();
    }

    private void clearShape(LayerGraph layer){
        int len = layer.prevLayerNames.size();
        layer.currentLayer.inputShape = new int[len][0];
        for (LayerGraph l : layer.nextLayer) {
            if (l.currentLayer.inputShape[0] != null) {
                clearShape(l);
            }
        }
    }

    private void resizeInput(){
        currentLayer.computeOutputShape();//run this layer
        for(int i = 0; i<currentLayer.featureMapInput.length; i++){
            currentLayer.featureMapInput[i] = null;
        }

        int[] os = currentLayer.outputShape;
        String shape1 = "Reshape layer "+ name + " output:";
        for(int i:os){
            shape1 += "," + i;
        }
        LogUtil.w("LayerGraph", shape1);

        if(nextLayer.size()==0){
            return;
        }

        for(LayerGraph next:nextLayer) {
            if(currentLayer instanceof Input){
                int[][] shape = next.currentLayer.inputShape;
                shape[0] = currentLayer.outputShape;
            }
            else {
                int[][] shape = next.currentLayer.inputShape;
                int index = getNextInputIndex(next);
                shape[index] = currentLayer.outputShape;
                next.currentLayer.featureMapInput[index] = currentLayer.featureMapOutput;
            }
        }

        for(LayerGraph next:nextLayer) {
            if(isAllInputShapeReady(next)){
                next.resizeInput();
            }
        }
    }

    public void init(){
        if(nextLayer.size()!=0){
            if(currentLayer instanceof Convolution && nextLayer.get(0).currentLayer instanceof ReLU){
                Convolution conv = (Convolution)currentLayer;
                conv.setNextRelu(true);
                ReLU relu = (ReLU) nextLayer.get(0).currentLayer;
                relu.setComputed(true);
            }
            if(currentLayer instanceof InnerProduct && nextLayer.get(0).currentLayer instanceof ReLU){
                InnerProduct fc = (InnerProduct)currentLayer;
                fc.setNextRelu(true);
                ReLU relu = (ReLU) nextLayer.get(0).currentLayer;
                relu.setComputed(true);
            }
        }

        currentLayer.computeOutputShape();//run this layer
        currentLayer.setup();
        currentLayer.setReady(true);
        for(int i = 0; i<currentLayer.featureMapInput.length; i++){
            currentLayer.featureMapInput[i] = null;
        }

        int[] os = currentLayer.outputShape;
        String shape1 = "layer "+ name + " output:";
        for(int i:os){
            shape1 += "," + i;
        }
        LogUtil.i("LayerGraph", shape1);

        if(nextLayer.size()==0){
            return;
        }

        for(LayerGraph next:nextLayer) {
            if(currentLayer instanceof Input){
                int[][] shape = next.currentLayer.inputShape;
                shape[0] = currentLayer.outputShape;
            }
            else {
                int[][] shape = next.currentLayer.inputShape;
                int index = getNextInputIndex(next);
                shape[index] = currentLayer.outputShape;
                next.currentLayer.featureMapInput[index] = currentLayer.featureMapOutput;
            }
        }

        for(LayerGraph next:nextLayer) {
            if(isAllInputShapeReady(next) && !next.currentLayer.isReady()){
                next.init();
            }
        }
    }
}

package com.rscnn.network;

import android.renderscript.Allocation;

public class FeatureMap {
    private int n;
    private int c;
    private int h;
    private int w;

    private boolean pad4 = false;
    private boolean matrix2D = true;

    private Allocation featureMap;

    public int getN() {
        return n;
    }

    public void setN(int n) {
        this.n = n;
    }

    public int getC() {
        return c;
    }

    public void setC(int c) {
        this.c = c;
    }

    public int getH() {
        return h;
    }

    public void setH(int h) {
        this.h = h;
    }

    public int getW() {
        return w;
    }

    public void setW(int w) {
        this.w = w;
    }

    public boolean isPad4() {
        return pad4;
    }

    public void setPad4(boolean pad4) {
        this.pad4 = pad4;
    }

    public boolean isMatrix2D() {
        return matrix2D;
    }

    public void setMatrix2D(boolean matrix2D) {
        this.matrix2D = matrix2D;
    }

    public Allocation getFeatureMap() {
        return featureMap;
    }

    public void setFeatureMap(Allocation featureMap) {
        this.featureMap = featureMap;
    }

    public String toString(){
        return "float["+n+"]["+h+"]["+w+"]["+c+"]";
    }

    public static float[][][][] copyFromAllocationVector4(Allocation allocation, int n, int h, int w, int c)
    {

        int channelAlign = c;

        int skip = 0;
        if(channelAlign % 4 != 0){
            channelAlign = c + 4 - c % 4;
            skip = 4 - c % 4;
        }
        float[] alloc = new float[n * h * w * channelAlign];
        float[][][][] output = new float[n][h][w][c];

        allocation.copyTo(alloc);
        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    for(int l=0;l<c;l++){
                        output[i][j][k][l] = alloc[count++];
                    }
                    count += skip;
                }
            }
        }
        return output;
    }

    private static float[][][][] copyFromAllocation(Allocation allocation, int n, int h, int w, int c)
    {

        float[] alloc = new float[n * h * w * c];
        float[][][][] output = new float[n][h][w][c];

        allocation.copyTo(alloc);
        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    for(int l=0;l<c;l++){
                        output[i][j][k][l] = alloc[count++];
                    }
                }
            }
        }
        return output;
    }

    private static float[][] copyFromAllocation2DVector4(Allocation allocation, int n, int c)
    {
        int ca = c;
        if(c % 4 !=0){
            ca = c + 4 - c % 4;
        }
        float[] alloc = new float[n * ca];
        float[][]output = new float[n][c];

        allocation.copyTo(alloc);
        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<ca;j++){
                if(j >= c){
                    count++;
                    continue;
                }
                output[i][j] = alloc[count++];
            }

        }
        return output;
    }

    private static float[][] copyFromAllocation2D(Allocation allocation, int n, int c)
    {

        float[] alloc = new float[n * c];
        float[][]output = new float[n][c];

        allocation.copyTo(alloc);
        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                output[i][j] = alloc[count++];
            }
        }
        return output;
    }

    private static void copyToAllocationVector4(float[][][][] input, Allocation allocation)
    {
        int n = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        int c = input[0][0][0].length;

        int channelAlign = c;
        int skip = 0;
        if(channelAlign % 4 !=0) {
            channelAlign = c + 4 - c % 4;
            skip = 4 - c % 4;
        }

        float[] output = new float[n * h * w * channelAlign];

        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    for(int l=0;l<c;l++){
                        output[count++] = input[i][j][k][l];
                    }
                    count += skip;
                }
            }
        }
        allocation.copyFrom(output);
    }

    private static void copyToAllocation(float[][][][] input, Allocation allocation)
    {
        int n = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        int c = input[0][0][0].length;

        float[] output = new float[n * h * w * c];


        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    for(int l=0;l<c;l++){
                        output[count++] = input[i][j][k][l];
                    }
                }
            }
        }
        allocation.copyFrom(output);
    }

    private static void copyToAllocation(float[][] input, Allocation allocation)
    {
        int n = input.length;
        int c = input[0].length;

        float[] output = new float[n * c];


        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                output[count++] = input[i][j];
            }
        }
        allocation.copyFrom(output);
    }

    public float[][][][] getData(){
        FeatureMap data = this;
        if(pad4){
            return copyFromAllocationVector4(data.getFeatureMap(), data.getN(),data.getH(),data.getW(),data.getC());
        }
        else{
            return copyFromAllocation(data.getFeatureMap(), data.getN(),data.getH(),data.getW(),data.getC());
        }
    }

    public float[] getData1D(){
        FeatureMap data = this;
        int size = data.getFeatureMap().getBytesSize() / 4 ;
        float[] output = new float[size];
        data.getFeatureMap().copyTo(output);
        return output;
    }

    public float[][] getData2D(){
        FeatureMap data = this;
        if(pad4){
            return copyFromAllocation2DVector4(data.getFeatureMap(), data.getN(),data.getC());
        }
        return copyFromAllocation2D(data.getFeatureMap(), data.getN(),data.getC());
    }

    public void setData(float[][] input){
        copyToAllocation(input, this.getFeatureMap());
        this.setN(input.length);
        this.setH(1);
        this.setW(1);
        this.setC(input[0].length);
    }
    public void setData(float[][][][] input){
        if(pad4) {
            copyToAllocationVector4(input, featureMap);
        }
        else{
            copyToAllocation(input, featureMap);
        }
        this.setN(input.length);
        this.setH(input[0].length);
        this.setW(input[0][0].length);
        this.setC(input[0][0][0].length);
    }

    public static float[][][][] transeposeToNCHW(float[][][][] input){
        int n = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        int c = input[0][0][0].length;

        float[][][][] output = new float[n][c][h][w];

        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<h;k++){
                    for(int l=0;l<w;l++){
                        output[i][j][k][l] = input[i][k][l][j];
                    }
                }
            }
        }
        return output;
    }

    public static float[][][][] transeposeToNHWC(float[][][][] input){
        int n = input.length;
        int c = input[0].length;
        int h = input[0][0].length;
        int w = input[0][0][0].length;

        float[][][][] output = new float[n][h][w][c];

        for(int i=0;i<n;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    for(int l=0;l<c;l++){
                        output[i][j][k][l] = input[i][l][j][k];
                    }
                }
            }
        }
        return output;
    }
}

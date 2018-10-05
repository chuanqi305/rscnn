package com.rscnn.algorithm;

import com.rscnn.utils.LogUtil;

public class ShapeUtils {
    public static void reshape1to4(float[] input, float[][][][] output){
        int n = output.length;
        int c = output[0].length;
        int h = output[0][0].length;
        int w = output[0][0][0].length;
        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<h;k++){
                    for(int l=0;l<w;l++){
                        output[i][j][k][l] = input[count++];
                    }
                }
            }
        }
    }

    public static float[][][][] reshape1to4(float[] input, int n, int c, int h, int w){
        float[][][][] output = new float[n][c][h][w];
        reshape1to4(input,output);
        return output;
    }

    public static void reshape4to1(float[][][][] input, float[] output){
        int n = input.length;
        int c = input[0].length;
        int h = input[0][0].length;
        int w = input[0][0][0].length;

        int count = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<h;k++){
                    for(int l=0;l<w;l++){
                        output[count++] = input[i][j][k][l];
                    }
                }
            }
        }
    }

    public static float[] reshape4to1(float[][][][] input){
        int n=input.length,c=input[0].length,h = input[0][0].length,w = input[0][0][0].length;
        float[] output = new float[n * c * h * w];
        reshape4to1(input,output);
        return output;
    }

    public static void reshape1to3(float[] input, float[][][] output){
        int c = output.length;
        int h = output[0].length;
        int w = output[0][0].length;
        int count = 0;
        for(int i=0;i<c;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    output[i][j][k] = input[count++];
                }
            }
        }
    }

    public static float[][][] reshape1to3(float[] input, int c, int h, int w){
        float[][][] output = new float[c][h][w];
        reshape1to3(input,output);
        return output;
    }

    public static void reshape3to1(float[][][] input, float[] output){
        int c = input.length;
        int h = input[0].length;
        int w = input[0][0].length;

        int count = 0;
        for(int i=0;i<c;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    output[count++] = input[i][j][k];
                }
            }
        }
    }

    public static float[] reshape3to1(float[][][] input){
        int c=input.length;
        int h = input[0].length;
        int w = input[0][0].length;

        float[] output = new float[c * h * w];
        reshape3to1(input,output);
        return output;
    }

    public static void reshape1to2(float[] input, float[][] output){
        int h = output.length;
        int w = output[0].length;
        int count = 0;
        for(int i=0;i<h;i++){
            for(int j=0;j<w;j++){
                output[i][j] = input[count++];
            }
        }
    }

    public static float[][] reshape1to2(float[] input, int h, int w){
        float[][] output = new float[h][w];
        reshape1to2(input,output);
        return output;
    }

    public static void reshape2to1(float[][] input, float[] output){
        int h = input.length;
        int w = input[0].length;
        int count = 0;
        for(int i=0;i<h;i++){
            for(int j=0;j<w;j++){
                output[count++] = input[i][j];
            }
        }
    }

    public static float[] reshape2to1(float[][] input){
        int h = input.length,w = input[0].length;
        float[] output = new float[h * w];
        reshape2to1(input,output);
        return output;
    }

    public static float[][] transpose2D(float[][] input){
        int len1d = input.length;
        int len2d = input[0].length;
        float[][] output = new float[len2d][len1d];
        for(int i=0;i<len1d;i++){
            for(int j=0;j<len2d;j++){
                output[j][i] = input[i][j];
            }
        }
        return output;
    }

    public static float[][][][] reshape4(float[][][][] in,int[] dim)
    {
        int n = in.length;
        int c = in[0].length;
        int h = in[0][0].length;
        int w = in[0][0][0].length;

        int n1 = dim[0];
        int c1 = dim[1];
        int h1 = dim[2];
        int w1 = dim[3];

        if(n1 * c1 * h1 * w1 != n * c  * h * w){
            LogUtil.e("ReshapeLayer", "reshape dim not match:input "+n+","+c+","+h+","+w
                    +";output is " +n1+","+c1+","+h1+","+w1);
            throw new IllegalArgumentException();
        }

        float[][][][] out = new float[n1][c1][h1][w1];
        int idn2 = 0,idc2 = 0,idh2 = 0,idw2 = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<h;k++){
                    for(int l=0;l<w;l++){
                        out[idn2][idc2][idh2][idw2] = in[i][j][k][l];
                        idw2++;
                        if(idw2>=w1) {
                            idw2 = 0;
                            idh2++;
                            if (idh2 >= h1) {
                                idh2 = 0;
                                idc2++;
                                if (idc2 >= c1) {
                                    idc2 = 0;
                                    idn2++;
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }


    public static void reshape2to1(short[][] input, short[] output){
        int h = input.length;
        int w = input[0].length;
        int count = 0;
        for(int i=0;i<h;i++){
            for(int j=0;j<w;j++){
                output[count++] = input[i][j];
            }
        }
    }

    public static short[] reshape2to1(short[][] input){
        int h = input.length,w = input[0].length;
        short[] output = new short[h * w];
        reshape2to1(input,output);
        return output;
    }
}

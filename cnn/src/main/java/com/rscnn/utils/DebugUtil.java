package com.rscnn.utils;

import android.os.Environment;

import com.rscnn.algorithm.ShapeUtils;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

public class DebugUtil {
    public static void dumpData(float[] data,String path){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] output = data;
        try {
            FileOutputStream fos = new FileOutputStream(base + path);
            for(int i=0;i<output.length;i++){
                //BigDecimal bigDecimal = new BigDecimal(output[i]);
                //fos.write(bigDecimal.toString().getBytes());
                fos.write(String.valueOf(output[i]).getBytes());
                if(i!=output.length-1){
                    fos.write(",".getBytes());
                }
            }
            fos.close();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
    public static void dumpData(float[][][][] data,String path){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] output = ShapeUtils.reshape4to1(data);
        try {
            FileOutputStream fos = new FileOutputStream(path);
            for(int i=0;i<output.length;i++){
                //BigDecimal bigDecimal = new BigDecimal(output[i]);
                //fos.write(bigDecimal.toString().getBytes());
                fos.write(String.valueOf(output[i]).getBytes());
                if(i!=output.length-1){
                    fos.write(",".getBytes());
                }
            }
            fos.close();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
    public static float[] parseFloatDataFromFile(String file){
        RandomAccessFile randomFile = null;
        try {
            randomFile = new RandomAccessFile(file, "r");
            byte[] buff = new byte[((int) randomFile.length())];
            randomFile.read(buff);
            String str = new String(buff);
            String num[] = str.split(",");
            int count = 0;
            float[] output = new float[num.length];
            for (int i = 0; i < num.length; i++) {
                output[i] = Float.parseFloat(num[count++]);
            }
            return output;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    public static int[] parseIntDataFromFile(String file){
        RandomAccessFile randomFile = null;
        try {
            randomFile = new RandomAccessFile(file, "r");
            byte[] buff = new byte[((int) randomFile.length())];
            randomFile.read(buff);
            String str = new String(buff);
            String num[] = str.split(",");
            int count = 1;
            int[] output = new int[num.length+1];
            for (int i = 0; i < output.length; i++) {
                output[i] = Integer.parseInt(num[count++]);
            }
            return output;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static float[][][][] parseBitmapFromFile(String file, int height, int width){
        float[][][][] arrayOfFloat = null;
        RandomAccessFile randomFile = null;
        try {
            arrayOfFloat = new float[1][3][height][width];
            randomFile = new RandomAccessFile(file, "r");
            byte[] buff = new byte[((int) randomFile.length())];
            randomFile.read(buff);
            String str = new String(buff);
            String num[] = str.split(",");
            int count = 1;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arrayOfFloat[0][0][i][j] = Float.parseFloat(num[count++]);
                    arrayOfFloat[0][1][i][j] = Float.parseFloat(num[count++]);
                    arrayOfFloat[0][2][i][j] = Float.parseFloat(num[count++]);
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayOfFloat;
    }

    public static void compareDataWithFile(float[] data, String filePath){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] data1 = parseFloatDataFromFile(base + filePath);
        int count=0;
        int fail = 0;
        int n = data.length;
        boolean isEqual = true;

        int toShow = 0;

        if(data1.length != n){
            LogUtil.e("DEBUG", "num shape does not match "+data1.length+"<==>"+n);
        }
        LogUtil.e("DEBUG", "start compare");
        for(int i=0;i<data.length;i++){
            float diff = Math.abs(data[i] - data1[count]);
            if(diff > 0.001f){
                LogUtil.e("DEBUG", "at["+i+"],("+data1[count]+") - ("+data[i]
                        +") = " + (data1[count]-data[i]));
                fail ++;
                isEqual = false;
                if(fail > 100){
                    return;
                }
            }
            else{
                if(toShow>0){
                    LogUtil.e("DEBUG", "at["+i+"],("+data1[count]+") - ("+data[i]
                            +") = " + (data1[count]-data[i]));
                    toShow--;
                }
            }
            count++;
        }
        if(isEqual) {
            LogUtil.e("DEBUG", "all data is equal");
        }
    }

    public static void compareDataWithFile(float[][][][] data, String filePath){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] data1 = parseFloatDataFromFile(base + filePath);
        int count=0;
        int fail = 0;
        int n = data.length;
        int c = data[0].length;
        int h = data[0][0].length;
        int w = data[0][0][0].length;
        boolean isEqual = true;

        int toShow = 0;

        if(data1.length != n*c*h*w){
            LogUtil.e("DEBUG", "num shape does not match "+data1.length+"<==>"+n*c*h*w);
        }
        LogUtil.e("DEBUG", "start compare");
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[0].length;j++){
                for(int k=0;k<data[0][0].length;k++){
                    for(int l=0;l<data[0][0][0].length;l++){
                        float diff = Math.abs(data[i][j][k][l] - data1[count]);
                        if(diff > 0.001f){
                            LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],("+data1[count]+") - ("+data[i][j][k][l]
                            +") = " + (data1[count]-data[i][j][k][l]));
                            fail ++;
                            isEqual = false;
                            if(fail > 100){
                                return;
                            }
                        }
                        else{
                            if(toShow>0){
                                LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],("+data1[count]+") - ("+data[i][j][k][l]
                                        +") = " + (data1[count]-data[i][j][k][l]));
                                toShow--;
                            }
                        }
                        count++;
                    }
                }
            }
        }
        if(isEqual) {
            LogUtil.e("DEBUG", "all data is equal");
        }
    }

    public static float[] parseFloatDataFromNumpyFile(String path){
        RandomAccessFile randomFile = null;
        try {
            randomFile = new RandomAccessFile(path, "r");
            byte[] buff = new byte[((int) randomFile.length())];
            randomFile.read(buff);
            float[] output = new float[buff.length / 4];
            int accum = 0;
            for(int i=0; i< buff.length; i++){
                int shift = i % 4;
                accum |= (buff[i] & 0xff) << i * 8;
                if(shift==3){
                    output[i / 4] = Float.intBitsToFloat(accum);
                    accum = 0;
                }
            }

            return output;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void compareData(float[][][][] data1, float[][][][] data2){
        int n = data1.length;
        int c = data1[0].length;
        int h = data1[0][0].length;
        int w = data1[0][0][0].length;

        int n1 = data2.length;
        int c1 = data2[0].length;
        int h1 = data2[0][0].length;
        int w1 = data2[0][0][0].length;

        int count=0;
        int fail = 0;
        int toShow = 0;

        if(n !=n1||c!=c1||h!=h1||w!=w1){
            LogUtil.e("DEBUG", "num shape does not match ");
        }
        boolean isEqual = true;
        LogUtil.e("DEBUG", "start compare");
        for(int i=0;i<data1.length;i++){
            for(int j=0;j<data1[0].length;j++){
                for(int k=0;k<data1[0][0].length;k++){
                    for(int l=0;l<data1[0][0][0].length;l++){
                        float diff = Math.abs(data1[i][j][k][l] - data2[i][j][k][l]);
                        if(diff > 0.0001f){
                            LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],(data1)("+data2[i][j][k][l]+") - ("+data1[i][j][k][l]
                                    +") = " + diff);
                            fail ++;
                            isEqual = false;
                            if(fail > 100){
                                return;
                            }
                        }
                        else{
                            if(toShow>0){
                                LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],(data1)("+data2[i][j][k][l]+") - ("+data1[i][j][k][l]
                                        +") = " + diff);
                                toShow--;
                            }
                        }
                        count++;
                    }
                }
            }
        }
        if(isEqual) {
            LogUtil.e("DEBUG", "all data is equal");
        }

    }

    public static void compareDataWithNumpyFile(float[][][][] data, String filePath){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] data1 = parseFloatDataFromNumpyFile(base + filePath);
        int count=0;
        int fail = 0;
        int n = data.length;
        int c = data[0].length;
        int h = data[0][0].length;
        int w = data[0][0][0].length;
        boolean isEqual = true;

        int toShow = 0;

        if(data1.length != n*c*h*w){
            LogUtil.e("DEBUG", "num shape does not match "+data1.length+"<==>"+n*c*h*w);
        }
        LogUtil.e("DEBUG", "start compare");
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[0].length;j++){
                for(int k=0;k<data[0][0].length;k++){
                    for(int l=0;l<data[0][0][0].length;l++){
                        float diff = Math.abs(data[i][j][k][l] - data1[count]);
                        if(diff > 0.0001f){
                            LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],(numpy)("+data1[count]+") - ("+data[i][j][k][l]
                                    +") = " + (data1[count] - data[i][j][k][l]));
                            fail ++;
                            isEqual = false;
                            if(fail > 100){
                                return;
                            }
                        }
                        else{
                            if(toShow>0){
                                LogUtil.e("DEBUG", "at["+i+"]["+j+"]["+k+"]["+l+"],("+data1[count]+") - ("+data[i][j][k][l]
                                        +") = " + (data1[count]-data[i][j][k][l]));
                                toShow--;
                            }
                        }
                        count++;
                    }
                }
            }
        }
        if(isEqual) {
            LogUtil.e("DEBUG", "all data is equal");
        }
    }
    public static void compareDataWithNumpyFile(float[] data, String filePath){
        String base = Environment.getExternalStorageDirectory().getPath() + "/";
        float[] data1 = parseFloatDataFromNumpyFile(base + filePath);
        int count=0;
        int fail = 0;
        int n = data.length;
        boolean isEqual = true;

        int toShow = 0;

        if(data1.length != n){
            LogUtil.e("DEBUG", "num shape does not match "+data1.length+"<==>"+n);
        }
        LogUtil.e("DEBUG", "start compare");
        for(int i=0;i<data.length;i++){
            float diff = Math.abs(data[i] - data1[count]);
            if(diff > 0.001f){
                LogUtil.e("DEBUG", "at["+i+"],("+data1[count]+") - ("+data[i]
                        +") = " + (data1[count]-data[i]));
                fail ++;
                isEqual = false;
                if(fail > 100){
                    return;
                }
            }
            else{
                if(toShow>0){
                    LogUtil.e("DEBUG", "at["+i+"],("+data1[count]+") - ("+data[i]
                            +") = " + (data1[count]-data[i]));
                    toShow--;
                }
            }
            count++;
        }
        if(isEqual) {
            LogUtil.e("DEBUG", "all data is equal");
        }
    }
}

package com.rscnn.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.util.Log;

import com.rscnn.network.DetectResult;

import java.text.DecimalFormat;
import java.util.List;

public class PhotoViewHelper {
    private final static int TEXT_SIZE = 70;
    private final static String TEXT_FONT = "DroidSans";
    private final static int TEXT_COLOR = Color.YELLOW;
    private final static int RECT_STROKE_WIDTH = 4;
    private final static int RECT_COLOR = Color.YELLOW;
    private final static int SCREEN_WIDTH = 1500;

    private static int[] zoomRect(float[] rect, float zoomRatio) {
        int[] newRect = new int[4];
        for(int i=0;i<4;i++){
            newRect[i] = (int)(rect[i] * zoomRatio);
        }
        return newRect;
    }

    public static Bitmap drawTextAndRect(Bitmap photo, List<DetectResult> detectResult) {
        int width = photo.getWidth();
        int height = photo.getHeight();
        float zoomRatio = (float)SCREEN_WIDTH/(float)width;
        int newHeight = (int)(height * zoomRatio);
        Bitmap bmp = Bitmap.createScaledBitmap(photo.copy(Bitmap.Config.ARGB_8888,true), SCREEN_WIDTH, newHeight, true);

        Canvas canvas = new Canvas(bmp);
        canvas.drawColor(Color.TRANSPARENT);

        Paint rectPaint = new Paint();

        rectPaint.setColor(RECT_COLOR);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(RECT_STROKE_WIDTH);

        Paint textPaint = new Paint();
        Typeface font = Typeface.create(TEXT_FONT, Typeface.NORMAL);
        textPaint.setTypeface(font);
        textPaint.setTextSize(TEXT_SIZE);
        textPaint.setColor(TEXT_COLOR);
        textPaint.setStyle(Paint.Style.FILL);

        DecimalFormat decimalFormat=new DecimalFormat("0.0");
        for(DetectResult res:detectResult) {
            float[] box = new float[]{res.getX1(),res.getY1(),res.getX2(),res.getY2()};
            int[] rect = zoomRect(box, zoomRatio);
            String prob = decimalFormat.format(res.getProb());
            String cls = res.getCls();
            Log.i("Draw", "class is "+cls+":"+prob);
            int textTop = rect[1]-15 > 0?rect[1]-10:rect[1] + 50;
            canvas.drawText(cls + ":" + prob, rect[0], textTop ,textPaint);
            canvas.drawRect(rect[0], rect[1], rect[2], rect[3], rectPaint);
        }
        canvas.save(Canvas.ALL_SAVE_FLAG);
        canvas.restore();
        return bmp;
    }
}

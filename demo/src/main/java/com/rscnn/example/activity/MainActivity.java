package com.rscnn.example.activity;

import android.content.ContentResolver;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.renderscript.RenderScript;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import com.rscnn.example.R;
import com.rscnn.model.MobileNetSSD;
import com.rscnn.model.ObjectDetector;
import com.rscnn.model.PvaLite;
import com.rscnn.network.ConvNet;
import com.rscnn.network.DetectResult;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import com.rscnn.utils.PhotoViewHelper;


public class MainActivity extends AppCompatActivity {

    private RenderScript rs;
    private ObjectDetector detector = null;
    private String modelPath = "mobilenet-ssd";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        rs = RenderScript.create(this);
        try {
            AssetManager assetManager = getAssets();
            String[] fileList = assetManager.list(modelPath);
            if (fileList.length != 0){
                detector = new MobileNetSSD(rs, assetManager, modelPath);
            }
            else {
                String modelDir = Environment.getExternalStorageDirectory().getPath() + "/" + modelPath;
                detector = new MobileNetSSD(rs, null, modelDir);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        setContentView(R.layout.activity_main);
    }

    public void btnClicked(View view) {
        Intent intent;
        intent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, 0);
    }

    private Bitmap cropImage(Bitmap image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int newHeight = height;
        int newWidth = width;
        int x = 0;
        int y = 0;
        if (height > width) {
            newHeight = width;
            y = (height - newHeight) / 2;
        }
        else {
            newWidth = height;
            x = (width - newWidth) / 2;
        }
        return Bitmap.createBitmap(image, x, y, newWidth, newHeight);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null) {
            return;
        }
        try {
            ContentResolver resolver = this.getContentResolver();
            Uri uri = data.getData();
            Bitmap bmp = MediaStore.Images.Media.getBitmap(resolver, uri);
            Bitmap image = cropImage(bmp);
            ImageView img = (ImageView) findViewById(R.id.imageView);
            List<DetectResult> result = detector.detect(image);
            Bitmap toDraw = PhotoViewHelper.drawTextAndRect(image, result);
            img.setImageBitmap(toDraw);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

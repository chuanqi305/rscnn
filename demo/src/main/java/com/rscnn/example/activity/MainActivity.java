package com.rscnn.example.activity;

import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
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
import java.util.List;
import com.rscnn.utils.PhotoViewHelper;


public class MainActivity extends AppCompatActivity {

    private RenderScript rs;
    private ObjectDetector detector = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        rs = RenderScript.create(this);
        try {
            detector = new PvaLite(rs, null, "/sdcard/pvalite");
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

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null) {
            return;
        }
        try {
            ContentResolver resolver = this.getContentResolver();
            Uri uri = data.getData();
            Bitmap bmp = MediaStore.Images.Media.getBitmap(resolver, uri);
            ImageView img = (ImageView) findViewById(R.id.imageView);
            List<DetectResult> result = detector.detect(bmp);
            Bitmap toDraw = PhotoViewHelper.drawTextAndRect(bmp, result);
            img.setImageBitmap(toDraw);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

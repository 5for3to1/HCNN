package com.lanytek.deepsensev3;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.squareup.picasso.Picasso;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageActivity extends AppCompatActivity {
    public static String TAG = "MobileNetSSD";

    private List<String> img_recognition_descriptions = new ArrayList<>();
//    private static final String [] mobileNetSSD_descriptions = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train","tvmonitor"};
    private static final String [] mobileNetSSD_descriptions = {"背景", "飞机", "自行车", "鸟", "船", "瓶子", "大巴车", "小汽车", "猫", "椅子", "牛", "桌子", "狗", "马", "摩托车", "人", "盆栽", "羊", "沙发","火车","显示器"};

    private Activity activity = this;

    private ImageView iv_image;
    private Button btn_loadModelGPU;
    private Button btn_processImage;
    private TextView tv_runtime;

    private static final int SELECT_PICTURE = 9999;
    private String selectedImagePath = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        iv_image = (ImageView) findViewById(R.id.iv_image);
        btn_loadModelGPU = (Button) findViewById(R.id.btn_loadModelGPU);
        btn_processImage = (Button) findViewById(R.id.btn_processImage);
        tv_runtime = (TextView) findViewById(R.id.tv_runTime);

        btn_loadModelGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new async_loadModel().execute();
            }
        });

        btn_processImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new async_processImage_mobilenetSSD().execute();
            }
        });

        iv_image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE);
            }
        });
    }

    private void setButtons(boolean isEnabled) {
        btn_loadModelGPU.setEnabled(isEnabled);
        btn_processImage.setEnabled(isEnabled);
    }

    private class async_loadModel extends AsyncTask<Void, Void, Void> {

        @Override
        protected void onPreExecute() {
            setButtons(false);
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... params) {

            NativeMethod.InitGPU("sdcard/MobileNetSSD-DeepSense-android", activity.getPackageName());

            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            setButtons(true);
        }
    }

    private class async_processImage_mobilenetSSD extends AsyncTask<Void, Void, Void> {

        private double t1,t2;
        private double cnn_runtime;
        private Bitmap bm = null;

        @Override
        protected void onPreExecute() {
            btn_processImage.setEnabled(false);
            tv_runtime.setText("------");
            t1 = System.currentTimeMillis();
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... params) {

            if(selectedImagePath != null) {
                final int IMG_X = 300;
                final int IMG_Y = 300;
                final int IMG_C = 3;

                final float [] bitmapArray = new float[IMG_X * IMG_Y * IMG_C];

                try {
                    bm = Picasso.with(activity)
                            .load(new File(selectedImagePath))
                            .config(Bitmap.Config.ARGB_8888)
                            .resize(IMG_X,IMG_Y)
                            .get();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(bm != null) {
                    for(int w = 0 ; w < bm.getWidth() ; w++) {
                        for(int h = 0 ; h < bm.getHeight() ; h++) {
                            int pixel = bm.getPixel(w, h);
                            for(int c = 0 ; c < 3 ; c++) {
                                bitmapArray[h * IMG_X * IMG_C + w * IMG_C + c] = Utilities.getColorPixel(pixel, c);
                            }
                        }
                    }
                }

                double x1 = System.currentTimeMillis();
                float [] result = NativeMethod.GetInferrence(bitmapArray);
                double x2 = System.currentTimeMillis();
                cnn_runtime = x2 - x1;
//                Log.d(TAG,"CNN RUNTIME: " + cnn_runtime + "ms");

                //do box drawing
                int image_width = iv_image.getDrawable().getBounds().width();
                int image_height = iv_image.getDrawable().getBounds().height();
                final Bitmap mutableBitmap = Bitmap.createScaledBitmap(bm, image_width, image_height, true).copy(bm.getConfig(), true);

                final Canvas canvas = new Canvas(mutableBitmap);

                for(int i = 0; i < result.length; i+=6)
                {
                    int label=(int) result[i];
                    float score = result[i+1];
                    int left=(int)(result[i+2]*image_width/IMG_X);
                    int top=(int)(result[i+3]*image_height/IMG_X);
                    int right=(int)(result[i+4]*image_width/IMG_X);
                    int bottom=(int)(result[i+5]*image_height/IMG_X);


                    if(left < 0 || left > mutableBitmap.getWidth() - 1)
                        left = 0;
                    if(right > mutableBitmap.getWidth() - 1 || right < 0)
                        right = mutableBitmap.getWidth() - 1;
                    if(top < 0 || top > mutableBitmap.getHeight() - 1)
                        top = 0;
                    if(bottom > mutableBitmap.getHeight() - 1 || bottom <0)
                        bottom = mutableBitmap.getHeight() - 1;
                    if(label >20 ||label <0)
                        label = 0;

                    Paint p = new Paint();
                    p.setStrokeWidth(4);
                    p.setColor(Color.RED);
                    p.setStyle(Paint.Style.STROKE);

                    Rect r = new Rect(left,top,right,bottom);
                    canvas.drawRect(r,p);

                    p.setTextSize(20f);
                    p.setColor(Color.BLUE);
                    p.setStyle(Paint.Style.FILL);
                    canvas.drawText("" + mobileNetSSD_descriptions[label],left,top-4,p);
                }

                activity.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        iv_image.setImageBitmap(mutableBitmap);
                    }
                });

            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            t2 = System.currentTimeMillis();
            double runtime = t2 - t1;
            btn_processImage.setEnabled(true);
            tv_runtime.setText(cnn_runtime + " ms");
        }
    }


    public String getPath(Uri uri) {
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = managedQuery(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }
    //选择图片，将图片置为空间背景
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                if(selectedImagePath != null)
                    iv_image.setImageURI(selectedImageUri);
            }
        }
    }

}

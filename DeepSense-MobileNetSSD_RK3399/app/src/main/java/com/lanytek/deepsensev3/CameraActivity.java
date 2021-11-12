package com.lanytek.deepsensev3;

import android.app.Activity;
import android.content.res.Configuration;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.os.Message;
import android.os.Bundle;
import android.util.Log;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import android.view.TextureView;
import android.view.SurfaceView;
import android.hardware.Camera;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Handler;

public class CameraActivity extends Activity implements Runnable,TextureView.SurfaceTextureListener
{
    private Handler handler;
    private final int MSG_1 = 123;
    private Thread thread;
    private boolean detection_flag;

    private SurfaceView sv_camera;
    private TextureView textureView;
    private Camera camera;

    final int IMG_X = 300;
    final int IMG_Y = 300;
    final int IMG_C = 3;
    //private static final String [] mobileNetSSD_descriptions = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train","tvmonitor"};
    private static final String [] mobileNetSSD_descriptions = {"背景", "飞机", "自行车", "鸟", "船", "瓶子", "大巴车", "小汽车", "猫", "椅子", "牛", "桌子", "狗", "马", "摩托车", "人", "盆栽", "羊", "沙发","火车","显示器"};
    private static final int [] object_color = {0xFFFFE000,0xFFF0F500,0xFFEFD500,0xFFE4B500,0xFFDAB900,0xFFD70000,0xFF450000,0xFF00FF00,0xFF0000,0xFAF0E600,0xF4A46000,0xF0FFFF00,0xE0FFFF00,0xD3D3D300,0xFFFFFF00,0x228B2200,0x00FFFF00,0xA52A2A00,0xDC143C00,0x70809000};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera);

        sv_camera = (SurfaceView)findViewById(R.id.sv_camera);
        textureView = (TextureView)findViewById(R.id.tv_camera);
        sv_camera.setZOrderOnTop(true);                             //至顶层
        sv_camera.getHolder().setFormat(PixelFormat.TRANSPARENT);   //透明

        init_camera();
        textureView.setSurfaceTextureListener(this);


        handler = new Handler(){
            @Override
            public void handleMessage(Message msg){
                super.handleMessage(msg);
                switch (msg.what)
                {
                    case MSG_1:
                        Canvas canvas = sv_camera.getHolder().lockCanvas();
                        canvas.drawColor(Color.WHITE);
                        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                        float [] detection_results=(float[])msg.obj;
                        for (int i=0;i<detection_results.length;i+=6)
                        {
                            int label=(int) detection_results[i];
                            float score = detection_results[i+1];
                            float scale_w = ((float)textureView.getWidth())/IMG_X;
                            float scale_h = ((float)textureView.getHeight())/IMG_Y;
                            int left=(int)(detection_results[i+2]*scale_w);
                            int top=(int)(detection_results[i+3]*scale_h);
                            int right=(int)(detection_results[i+4]*scale_w);
                            int bottom=(int)(detection_results[i+5]*scale_h);

                            Paint paint = new Paint();
                            paint.setColor(Color.RED);
                            paint.setStyle(Paint.Style.STROKE);
                            paint.setStrokeWidth(8);
                            Rect r = new Rect(left,top,right,bottom);

                            canvas.drawRect(r,paint);
                            paint.setTextSize(40f);
//                            paint.setColor(Color.BLUE);
                            paint.setColor(object_color[label-1]);
                            paint.setStyle(Paint.Style.FILL);
                            canvas.drawText("" + mobileNetSSD_descriptions[label],left,top-8,paint);
                        }
                        sv_camera.getHolder().unlockCanvasAndPost(canvas);
                        break;
                    default:
                        break;
                }
            }
        };

        detection_flag=true;
        thread = new Thread(this);
        thread.start();
    }

    private void init_camera() {
        Log.e("","测试");
        int numberOfCameras = Camera.getNumberOfCameras();// 获取摄像头个数
        if(numberOfCameras<1){
            Toast.makeText(this,"no camera",Toast.LENGTH_LONG).show();
            finish();
            return;
        }
    }

    @Override
    public void run() {
        NativeMethod.InitGPU("sdcard/MobileNetSSD-DeepSense-android",this.getPackageName());
        while(detection_flag)
        {
            Bitmap bmp = textureView.getBitmap();
            if(bmp != null)
            {
                int bitmnap_w=bmp.getWidth();
                int bitmnap_h=bmp.getHeight();
                Matrix matrix=new Matrix();
                float scaleWidth = ((float)IMG_X)/bitmnap_w;
                float scaleHeight = ((float)IMG_Y)/bitmnap_h;
                matrix.postScale(scaleWidth,scaleHeight);
                Bitmap bmp_input=Bitmap.createBitmap(bmp,0,0,bitmnap_w,bitmnap_h,matrix,true);

                final float [] bitmapArray = new float[IMG_X * IMG_Y * IMG_C];
                if(bmp_input != null) {
                    for(int w = 0 ; w < bmp_input.getWidth() ; w++) {
                        for(int h = 0 ; h < bmp_input.getHeight() ; h++) {
                            int pixel = bmp_input.getPixel(w, h);
                            for(int c = 0 ; c < 3 ; c++) {
                                bitmapArray[h * IMG_X * IMG_C + w * IMG_C + c] = Utilities.getColorPixel(pixel, c);
                            }
                        }
                    }

                    float[] result = NativeMethod.GetInferrence(bitmapArray);
                    Message message = new Message();
                    message.what = MSG_1;
                    message.obj = result;
                    handler.sendMessage(message);
                }
            }
        }
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {

        int camera_id=0;
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        int numberOfCameras = Camera.getNumberOfCameras();
        for (int i = 0; i < numberOfCameras; i++)
        {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK)
            {
                camera_id = i;
            }
        }
        camera = Camera.open(camera_id);
        if(camera != null)
        {
            Camera.Parameters parameters = camera.getParameters();
            int previewWidth=width,previewHeight=height;
            List<Camera.Size> sizeList = parameters.getSupportedPreviewSizes();
            if(sizeList.size() > 1)
            {
                Iterator<Camera.Size> iterable=sizeList.iterator();
                while (iterable.hasNext()) {
                    Camera.Size cur = iterable.next();
                    previewWidth = cur.width;
                    previewHeight = cur.height;
                    if(this.getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE){
                        if (previewWidth >= height && previewHeight >= width) {
                            break;
                        }
                    }
                    else
                    {
                        if (previewWidth >= width && previewHeight >= height) {
                            break;
                        }
                    }

                }
                parameters.setPreviewSize(previewWidth,previewHeight);
                parameters.setPictureSize(previewWidth,previewHeight);
            }
            // 设置自动对焦模式
            List<String> focusModes = parameters.getSupportedFocusModes();
            if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO))
            {
                parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
            }
//            parameters.setPreviewFrameRate(20);
            parameters.setPictureFormat(PixelFormat.JPEG);
            parameters.set("jpeg-quality",85);
            camera.setParameters(parameters);
            try {
                if(this.getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE)
                {
                    camera.setDisplayOrientation(90);       //设置预览角度，并不改变获取到的原始数据方向
                }
                else
                {
                    camera.setDisplayOrientation(0);        //设置预览角度，并不改变获取到的原始数据方向
                }
                // 绑定相机和预览的View
                camera.setPreviewTexture(surface);
                // 开始预览
                camera.startPreview();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        camera.stopPreview();
        camera.release();
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        detection_flag=false;
        if(null != camera)
        {
            camera.stopPreview();
            camera.release();
        }
    }
}

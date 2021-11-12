package com.lanytek.deepsensev3;

import android.app.Activity;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Handler;
import android.os.Message;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class FaceActivity extends Activity implements Runnable,TextureView.SurfaceTextureListener{

    private Handler handler;
    private final int MSG_1 = 123;
    private Thread thread;
    private boolean detection_flag;

    private SurfaceView sv_face;
    private TextureView textureView;
    private Camera camera;

    int textureView_w;
    int textureView_h;
    final int IMG_X = 300;
    final int IMG_Y = 300;
    final int IMG_C = 3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_face);

        sv_face = (SurfaceView)findViewById(R.id.sv_face);
        textureView = (TextureView)findViewById(R.id.tv_face);
        sv_face.setZOrderOnTop(true);                             //至顶层
        sv_face.getHolder().setFormat(PixelFormat.TRANSPARENT);   //透明

        init_camera();
        textureView.setSurfaceTextureListener(this);

        handler = new Handler(){
            @Override
            public void handleMessage(Message msg){
                super.handleMessage(msg);
                switch (msg.what)
                {
                    case MSG_1:
                        Canvas canvas = sv_face.getHolder().lockCanvas();
                        canvas.drawColor(Color.WHITE);
                        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                        float scale_w, scale_h;
//                        int x_offset,y_offset;
//                        if(textureView_w < textureView_h)
//                        {
//                            scale_w = ((float)textureView_w)/IMG_X;
//                            scale_h = ((float)textureView_w)/IMG_Y;
//                            x_offset = 0;
//                            y_offset = (textureView_h-textureView_w)/2;
//                        }
//                        else
//                        {
//                            scale_w = ((float)textureView_h)/IMG_X;
//                            scale_h = ((float)textureView_h)/IMG_Y;
//                            x_offset = (textureView_w-textureView_h)/2;
//                            y_offset = 0;
//                        }
                        scale_w = ((float)textureView_w)/IMG_X;
                        scale_h = ((float)textureView_h)/IMG_Y;

                        float [] detection_results=(float[])msg.obj;
                        for (int i=0;i<detection_results.length;i+=6)
                        {
                            int label=(int) detection_results[i];
                            float score = detection_results[i+1];
                            int left=(int)(detection_results[i+2]*scale_w);
                            int top=(int)(detection_results[i+3]*scale_h);
                            int right=(int)(detection_results[i+4]*scale_w);
                            int bottom=(int)(detection_results[i+5]*scale_h);

                            Paint paint = new Paint();
                            paint.setColor(Color.RED);
                            paint.setStyle(Paint.Style.STROKE);
                            paint.setStrokeWidth(8);
                            Rect r = new Rect(left,top ,right,bottom);

                            canvas.drawRect(r,paint);
                            paint.setTextSize(40f);
                            paint.setColor(Color.BLUE);
                            paint.setStyle(Paint.Style.FILL);
                            canvas.drawText("" + score,left,top-8,paint);
                        }
                        sv_face.getHolder().unlockCanvasAndPost(canvas);
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

    @Override
    public void run() {
        NativeMethod.InitGPU("sdcard/MobileNetSSD-face-DeepSense-android",this.getPackageName());
        textureView_w = textureView.getWidth();
        textureView_h = textureView.getHeight();
        while(detection_flag)
        {
            Bitmap bmp = textureView.getBitmap();
            if(bmp != null)
            {
//                int bitmnap_w=bmp.getWidth();
//                int bitmnap_h=bmp.getHeight();
//                //先裁剪
//                Bitmap bmp_cut;
//                if(bitmnap_h > bitmnap_w)
//                {
//                    int y_offset = (bitmnap_h-bitmnap_w)/2;
//                    bmp_cut = Bitmap.createBitmap(bmp,0,y_offset,bitmnap_w,bitmnap_w);
//                }
//                else
//                {
//                    int x_offset = (bitmnap_w-bitmnap_h)/2;
//                    bmp_cut = Bitmap.createBitmap(bmp,x_offset,0,bitmnap_h,bitmnap_h);
//                }
//                //后缩放
//                Bitmap bmp_input = Bitmap.createScaledBitmap(bmp_cut,IMG_X,IMG_Y,true);
                //直接缩放
                Bitmap bmp_input = Bitmap.createScaledBitmap(bmp,IMG_X,IMG_Y,true);

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
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {

        int camera_id=0;
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        int numberOfCameras = Camera.getNumberOfCameras();
        for (int i = 0; i < numberOfCameras; i++)
        {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT)
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
                    camera.setDisplayOrientation(90);       //设置预览角度，旋转90度
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
    public void onBackPressed() {
        super.onBackPressed();
        detection_flag=false;
        NativeMethod.ReleaseCNN();
        if(null != camera)
        {
            camera.stopPreview();
            camera.release();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(null != camera)
        {
            camera.stopPreview();
            camera.release();
        }
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

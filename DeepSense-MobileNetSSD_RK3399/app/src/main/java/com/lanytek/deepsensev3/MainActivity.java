package com.lanytek.deepsensev3;

import android.app.Activity;
import android.content.Intent;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;


public class MainActivity extends AppCompatActivity {

    private Button btn_image_detection;
    private Button btn_camera_detection;
    private Button btn_face_detection;

    private Activity activity = this;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn_image_detection=(Button)findViewById(R.id.btn_image);
        btn_camera_detection=(Button)findViewById(R.id.btn_camera);
        btn_face_detection=(Button)findViewById(R.id.btn_face);

        new async_copy_kernel_code().execute("deepsense.cl");

        btn_image_detection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent();
                intent.setClass(MainActivity.this,ImageActivity.class);
                startActivity(intent);
            }
        });

        btn_camera_detection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent();
                intent.setClass(MainActivity.this,CameraActivity.class);
                startActivity(intent);
            }
        });

        btn_face_detection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent();
                intent.setClass(MainActivity.this,FaceActivity.class);
                startActivity(intent);
            }
        });
    }

    private void setButtons(boolean isEnabled) {
        btn_image_detection.setEnabled(isEnabled);
        btn_camera_detection.setEnabled(isEnabled);
        btn_face_detection.setEnabled(isEnabled);
    }

    //设备没有root权限，将文件写进execdir
    private class async_copy_kernel_code extends AsyncTask<String, Void, Void> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            setButtons(false);
        }

        @Override
        protected Void doInBackground(String... params) {
            for(String p : params) {
                Utilities.copyFile(activity, p);
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            setButtons(true);
        }
    }
}

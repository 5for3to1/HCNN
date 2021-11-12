package com.lanytek.deepsensev3;

import android.app.Activity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Created by JC1DA on 3/16/16.
 */
public class Utilities {

    public static float getColorPixel(int pixel, int color) {
        float value = 0;

        switch (color) {
//            case 0:
//                value = (float)((pixel >> 16) & 0x000000ff);
//                value = (float)((value-127.5)*0.007843);
//                break;
//            case 1:
//                value = (float)((pixel >> 8) & 0x000000ff);
//                value = (float)((value-127.5)*0.007843);
//                break;
//            case 2:
//                value = (float)(pixel & 0x000000ff);
//                value = (float)((value-127.5)*0.007843);
//                break;

            case 0:
                value = (float)((pixel >> 0) & 0x000000ff);
                value = (float)((value-127.5)*0.007843);
                break;
            case 1:
                value = (float)((pixel >> 8) & 0x000000ff);
                value = (float)((value-127.5)*0.007843);
                break;
            case 2:
                value = (float)((pixel >> 16) & 0x000000ff);
                value = (float)((value-127.5)*0.007843);
                break;
        }

        return value;
    }

    public static byte getColorPixelInt8(int pixel, int color) {
        byte value = 0;

        switch (color) {
            case 0:
                value = (byte)((pixel >> 16) & 0x000000ff);
                break;
            case 1:
                value = (byte)((pixel >> 8) & 0x000000ff);
                break;
            case 2:
                value = (byte)(pixel & 0x000000ff);
                break;
        }

        return value;
    }

    public static void copyFile(Activity activity, final String f) {
        InputStream in;
        try {
            in = activity.getAssets().open(f);
            final File of = new File(activity.getDir("execdir", activity.MODE_PRIVATE), f);

            final OutputStream out = new FileOutputStream(of);

            final byte b[] = new byte[65535];
            int sz = 0;
            while ((sz = in.read(b)) > 0) {
                out.write(b, 0, sz);
            }
            in.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

package com.lanytek.deepsensev3;

public class NativeMethod {

    // Used to load the 'native-lib' library
    static {
        System.loadLibrary("deepsense");
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native void InitGPU(String model_dir_path, String packageName);
    public static native float [] GetInferrence(float[] input);
    public static native void ReleaseCNN();
}

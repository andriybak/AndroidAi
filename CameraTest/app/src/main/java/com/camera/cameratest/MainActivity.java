package com.camera.cameratest;

import android.gesture.Prediction;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.BitmapDrawable;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.MainThread;
import android.view.Surface;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.app.Activity;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Size;
import android.view.Display;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends Activity {
    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov2-tiny-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;
    private static final int cropSize = YOLO_INPUT_SIZE;

    private ClassesParser m_ClassesToNames;

    private TensorFlowInferenceInterface tf;
    SurfaceView sv;
    SurfaceHolder m_holder;
    HolderCallback holderCallback;
    Camera camera;
    TextView text;
    ImageView image;
    Bitmap testImage;
    private Classifier detector;

    boolean CameraIsLocked = false;

    static
    {
        System.loadLibrary("tensorflow_inference");
    }

    private String MODEL_PATH = "file:///android_asset/mobilenet_v2_1.4_224_frozen.pb";

    final int CAMERA_ID = 0;
    final boolean FULL_SCREEN = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);
        text = findViewById(R.id.textView2);
        sv = (SurfaceView) findViewById(R.id.surfaceView);
        image =  findViewById(R.id.imageView);
        m_holder = sv.getHolder();
        m_holder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        holderCallback = new HolderCallback();
        m_holder.addCallback(holderCallback);


        m_ClassesToNames = new ClassesParser(this.getAssets());


        testImage = ((BitmapDrawable)this.getResources().getDrawable(R.drawable.zara2)).getBitmap();

        tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);

        Bitmap m_bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.zara2);

        Bitmap largeIcon = Bitmap.createScaledBitmap(m_bitmap, cropSize, cropSize, false);

        detector = TensorFlowYoloDetector.create(getAssets(), YOLO_MODEL_FILE, YOLO_INPUT_SIZE, YOLO_INPUT_NAME, YOLO_OUTPUT_NAMES, YOLO_BLOCK_SIZE);

        final List<Classifier.Recognition> results = detector.recognizeImage(largeIcon);

        for(int ii = 0; ii < results.size(); ii++)
        {
            Classifier.Recognition item = results.get(ii);
            if(item.getConfidence() > 0.5)
            {
                text.setText(text.getText() + "\n" + item.getTitle());
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        camera = Camera.open(CAMERA_ID);
        setPreviewSize(FULL_SCREEN);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (camera != null)
            camera.release();
        camera = null;
    }

    public class HolderCallback implements SurfaceHolder.Callback {

        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            try {
                camera.setPreviewDisplay(holder);
                camera.startPreview();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void surfaceChanged(final SurfaceHolder holder, int format, int width,
                                   int height) {
            camera.stopPreview();
            setCameraDisplayOrientation(CAMERA_ID);
            try {

                camera.setPreviewDisplay(holder);
                camera.stopPreview();

                camera.setPreviewCallback(new Camera.PreviewCallback() {
                    @Override
                    public void onPreviewFrame(byte[] bytes, Camera camera) {
                        if(CameraIsLocked == false)
                        {
                            new PredictionTask().execute(bytes, camera);
                            CameraIsLocked = true;
                        }
                    }
                });

                camera.startPreview();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
        }
    }

    private List<Classifier.Recognition> ResizeBoxestoCameraSize(List<Classifier.Recognition> dogboxes, float ToWidth, float ToHeight, float currentWidth, float currentHeight)
    {
        List<Classifier.Recognition> resizedBoxes = dogboxes;
        float refWidth = ((float)currentWidth)/(ToWidth);
        float refHeight = ((float)currentHeight)/(ToHeight);

        for(int ii = 0; ii < resizedBoxes.size(); ii++)
        {
            Classifier.Recognition value = resizedBoxes.get(ii);
            RectF resizedBox = new RectF(value.getLocation().left/refWidth, value.getLocation().top/refHeight, value.getLocation().right/refWidth, value.getLocation().bottom/refHeight);
            value.setLocation(resizedBox);
        }

        return resizedBoxes;
    }

    private ArrayList<Classifier.Recognition> onlyDogsWithTreshHold(List<Classifier.Recognition> boxes, double treshHold)
    {
        ArrayList<Classifier.Recognition> result = new ArrayList<>();
        for(int ii = 0 ; ii < boxes.size(); ii++)
        {
           if(boxes.get(ii).getConfidence() > treshHold && (boxes.get(ii).getTitle() == "dog" || boxes.get(ii).getTitle() == "cat"))
           {
               result.add(boxes.get(ii));
           }
        }

        return result;
    }

    public List<String> DoRecognition(Bitmap yoloResizedPicture, List<Classifier.Recognition> dogBoxes, double breedTreshHold)
    {

        ArrayList<String> resultBreeds = new ArrayList<String>();
        for(int ii = 0; ii < dogBoxes.size(); ii++)
        {
            RectF box = dogBoxes.get(ii).getLocation();
            int boxWidth = (int)(box.right - box.left);
            int boxHeight = (int)(box.bottom - box.top);

            Bitmap cropImage = Bitmap.createBitmap(yoloResizedPicture, (int)box.left, (int)box.top, boxWidth, boxHeight);
            Bitmap m_bitmapForNn = cropImage.createScaledBitmap(cropImage, 224, 224, false);

            float[] resultMobileNets = PredictWithMobileNets(m_bitmapForNn);
            // ArrayList<Integer> bestIndexes = new ArrayList<Integer>();
            double minValue = breedTreshHold;
            for(int jj = 0; jj<resultMobileNets.length;jj++)
            {
                if(resultMobileNets[jj] > minValue)
                {
                    if(resultBreeds.size() > ii)
                    {
                       String currentValue = resultBreeds.get(ii);
                       resultBreeds.set(ii, currentValue + "\n" +m_ClassesToNames.GetValueByIndex(new Integer(jj)));
                    }
                    else
                    {
                        resultBreeds.add(m_ClassesToNames.GetValueByIndex(new Integer(jj)));
                    }
                    //bestIndexes.add(new Integer(ii));
                }
            }
        }


        Collections.sort(resultBreeds);
     /*   if(resultBreeds.size() > 0)
            text.setText(resultBreeds.get(0));
*/
     /*   for(int ii = 0; ii < bestIndexes.size(); ii++)
        {
            Integer index = bestIndexes.get(ii);
            text.setTextColor(Color.WHITE);
            String breedName = m_ClassesToNames.GetValueByIndex(index);
            resultBreeds.add(breedName);
            text.setText(text.getText() + "\n" + breedName);
        }
*/
        return resultBreeds;
    }

    void setPreviewSize(boolean fullScreen) {

        Display display = getWindowManager().getDefaultDisplay();
        boolean widthIsMax = display.getWidth() > display.getHeight();

        Size size = camera.getParameters().getPreviewSize();

        RectF rectDisplay = new RectF();
        RectF rectPreview = new RectF();

        rectDisplay.set(0, 0, display.getWidth(), display.getHeight());

        if (widthIsMax) {
            rectPreview.set(0, 0, size.width, size.height);
        } else {
            rectPreview.set(0, 0, size.height, size.width);
        }

        Matrix matrix = new Matrix();
        if (!fullScreen) {
            matrix.setRectToRect(rectPreview, rectDisplay,
                    Matrix.ScaleToFit.START);
        } else {
            matrix.setRectToRect(rectDisplay, rectPreview,
                    Matrix.ScaleToFit.START);
            matrix.invert(matrix);
        }

        matrix.mapRect(rectPreview);

        sv.getLayoutParams().height = (int) (rectPreview.bottom);
        sv.getLayoutParams().width = (int) (rectPreview.right);
    }

    void setCameraDisplayOrientation(int cameraId) {
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        int result = 0;

        CameraInfo info = new CameraInfo();
        Camera.getCameraInfo(cameraId, info);

        if (info.facing == CameraInfo.CAMERA_FACING_BACK) {
            result = ((360 - degrees) + info.orientation);
        } else
            if (info.facing == CameraInfo.CAMERA_FACING_FRONT) {
                result = ((360 - degrees) - info.orientation);
                result += 360;
            }
        result = result % 360;
        camera.setDisplayOrientation(result);
    }

    public float[] PredictWithMobileNets(Bitmap bitmapForNM)
    {
        int m_nImageSize = 224;
        String INPUT_NAME = "input";
        String OUTPUT_NAME = "MobilenetV2/Predictions/Reshape_1";

        float[] m_arrInput = new float[
                m_nImageSize * m_nImageSize * 3];
        int[] intValues = new int[
                m_nImageSize * m_nImageSize];

        bitmapForNM.getPixels(intValues, 0,
                m_nImageSize, 0, 0, m_nImageSize,
                m_nImageSize);

        for (int i = 0; i < intValues.length; i++)
        {
            int val = intValues[i];
            m_arrInput[i * 3 + 0] =
                    ((val >> 16) & 0xFF) / 255f;
            m_arrInput[i * 3 + 1] =
                    ((val >> 8) & 0xFF) / 255f;
            m_arrInput[i * 3 + 2] =
                    (val & 0xFF) / 255f;
        }

        tf.feed(INPUT_NAME, m_arrInput, 1,
                m_nImageSize, m_nImageSize, 3);

        tf.run(new String[]{OUTPUT_NAME}, false);

        float[] m_arrPrediction = new float[4004];
        tf.fetch(OUTPUT_NAME, m_arrPrediction);

        return  m_arrPrediction;
    }



    //THREAD TO PREDICT

    class PredictionTask extends
            AsyncTask<Object, Void, Void>
    {
        @Override
        protected void onPreExecute()
        {
            super.onPreExecute();
        }

        // ---

        @Override
        protected Void doInBackground(Object ... params)
        {
            try
            {
                byte[] bytes = (byte[])params[0];
                Camera camera = (Camera)params[1];
                Camera.Parameters parameters = camera.getParameters();
                int width = parameters.getPreviewSize().width;
                int height = parameters.getPreviewSize().height;

                YuvImage yuv = new YuvImage(bytes, parameters.getPreviewFormat(), width, height, null);
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                yuv.compressToJpeg(new Rect(0, 0, width, height), 50, out);
                byte[] data = out.toByteArray();
                Bitmap cameraBitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
                if(cameraBitmap != null)
                {
                    final Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    cameraBitmap = Bitmap.createBitmap(cameraBitmap, 0, 0, cameraBitmap.getWidth(), cameraBitmap.getHeight(),matrix, false);
                    //Bitmap m_bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.zara2);
                    Bitmap scaledYOLOCamera = Bitmap.createScaledBitmap(cameraBitmap, cropSize, cropSize, false);
                    final List<Classifier.Recognition> results = detector.recognizeImage(scaledYOLOCamera);

                    if (results.size() > 0)
                    {
                        //Only dog boxes with accuracy at 0.01
                        final List<Classifier.Recognition> resultDogBoxes = onlyDogsWithTreshHold(results, 0.6);
                        if (resultDogBoxes.size() > 0) {
                            final List<String> resultDogBreeds = DoRecognition(scaledYOLOCamera, resultDogBoxes, 0.3);
                            if(resultDogBreeds.size() > 0)
                            {
                                final List<Classifier.Recognition> scaledDogBoxes = ResizeBoxestoCameraSize(resultDogBoxes, cameraBitmap.getWidth(), cameraBitmap.getHeight(), cropSize, cropSize);
                                final Bitmap bitmap = Bitmap.createBitmap(cameraBitmap.getWidth(), cameraBitmap.getHeight(), Bitmap.Config.ARGB_8888);

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        try{
                                            DrawView drawView = new DrawView(getApplicationContext(), scaledDogBoxes, resultDogBreeds);
                                            Canvas canvas = new Canvas(bitmap);
                                            ((View) drawView).draw(canvas);
                                            ImageView iv = findViewById(R.id.imageView);
                                            iv.setImageBitmap(bitmap);
                                            text.setTextColor(Color.WHITE);
                                            text.setText(resultDogBreeds.get(0));
                                        }
                                        catch (Exception e)
                                        {
                                            String message = e.getMessage();
                                        }
                                    }
                                });
                            }
                        }
                        else
                        {
                            CameraIsLocked = false;
                        }
                    }
                    else
                    {
                        CameraIsLocked = false;
                    }
                }
            }
            catch (Exception e)
            {
                    e.getMessage();
                    CameraIsLocked = false;
            }
            return null;
        }

        // ---

        @Override
        protected void onPostExecute(Void result)
        {
            super.onPostExecute(result);
            CameraIsLocked = false;
        }
    }
}


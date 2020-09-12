package com.dogbreed.detector;

import android.content.res.AssetManager;

import android.graphics.Bitmap;
import android.graphics.RectF;

import android.os.Build;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

public class DogBreedClassifier implements Classifier
{
    private static final Logger LOGGER = new Logger();

    private static HashMap<Integer, String> indexToCode = new HashMap<Integer, String>();
    private static HashMap<String, String> codesToHuman = new HashMap<String, String>();

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private static boolean isNNAPI = false;
    private static boolean isGPU = true;

    // config yolov4 tiny
    private static final int INPUT_SIZE = 416;
    private static final int[] OUTPUT_WIDTH_TINY = new int[]{2535, 2535};

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();

    private ByteBuffer imgData;

    private Interpreter yoloIdentifier;
    private Interpreter mobileNetLite;

    private static ArrayList<Integer> possibleDetectionClasses = new ArrayList<>();
    private final static int[] POSSIBLE_DOG_CLASSES_ARRAY = { 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77 };

    //Class constructor
    private DogBreedClassifier() { }

    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename)
            throws IOException
    {
        final DogBreedClassifier dogBreedClassifier = new DogBreedClassifier();

        final String[] splitName = labelFilename.split("file:///android_asset/");
        String actualFilename = "";
        if (splitName.length > 1)
        {
            actualFilename = splitName[1];
        }

        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null)
        {
            LOGGER.w(line);
            dogBreedClassifier.labels.add(line);
        }

        br.close();

        try
        {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(false);
                    options.setAllowFp16PrecisionForFp32(true);
                    options.setAllowBufferHandleOutput(true);
                    options.setUseNNAPI(true);
                }
            }

            dogBreedClassifier.mobileNetLite = new Interpreter(Utils.loadModelFile(assetManager,"v3_float.tflite"), options);

            if (isGPU) {
                GpuDelegate gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            }

            dogBreedClassifier.yoloIdentifier = new Interpreter(Utils.loadModelFile(assetManager, modelFilename), options);

            // Pre-allocate buffers.
            final int numBytesPerChannel = 4;
            dogBreedClassifier.imgData = ByteBuffer.allocateDirect(1 * dogBreedClassifier.INPUT_SIZE * dogBreedClassifier.INPUT_SIZE * 3 * numBytesPerChannel);
            dogBreedClassifier.imgData.order(ByteOrder.nativeOrder());

            possibleDetectionClasses = FillPossibleClassesList();
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }

        InitializeMap(assetManager);

        return dogBreedClassifier;
    }

    private static ArrayList<Integer> FillPossibleClassesList()
    {
        ArrayList<Integer> allClasses = new ArrayList<>();
        for (int number : POSSIBLE_DOG_CLASSES_ARRAY)
        {
            allClasses.add(new Integer(number));
        }

        return  allClasses;
    }

    private static void InitializeMap(AssetManager assets)
    {
        ReadCodesFile(assets);
        FillCodesWithValues(assets);
    }

    private static ArrayList<String> getDogCodesFromFile(AssetManager assets)
    {
        ArrayList<String> dogCodes = new ArrayList<>();
        BufferedReader reader = null;
        try
        {
            reader = new BufferedReader(new InputStreamReader(assets.open("dog_codes.txt")));
            String mLine;
            while ((mLine = reader.readLine()) != null)
            {
                dogCodes.add(mLine.trim());
            }
        }
        catch (IOException e) {
            //log the exception
        }

        return dogCodes;
    }

    private static void FillCodesWithValues(AssetManager assets)
    {
        ArrayList<String> onlyDogCodes = getDogCodesFromFile(assets);
        BufferedReader reader = null;
        try
        {
            reader = new BufferedReader(new InputStreamReader(assets.open("dog_code_names.txt")));
            String mLine;
            while ((mLine = reader.readLine()) != null)
            {
                String[] codeAndName = mLine.trim().split("\t");
                if(codeAndName.length >= 2 && onlyDogCodes.contains(codeAndName[0]))
                {
                    codesToHuman.put(codeAndName[0], codeAndName[1]);
                }
            }
        }
        catch (IOException e) {
            //log the exception
        }
        finally
        {
            if (reader != null)
            {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    private static void ReadCodesFile(AssetManager assets)
    {
        BufferedReader reader = null;
        try
        {
            reader = new BufferedReader(
                    new InputStreamReader(assets.open("codes.txt")));

            String mLine;
            int index = 1;
            while ((mLine = reader.readLine()) != null) {
                //process line
                mLine.trim();
                indexToCode.put(index, mLine);
                index++;
            }
        }
        catch (IOException e) {
            //log the exception
        }
        finally
        {
            if (reader != null)
            {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    public void setNumThreads(int num_threads)
    {
        if (yoloIdentifier != null) yoloIdentifier.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked)
    {
        if (yoloIdentifier != null) yoloIdentifier.setUseNNAPI(isChecked);
    }

    @Override
    public float getObjThresh()
    {
        return DetectorActivity.MINIMUM_CONFIDENCE_SCORE;
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(final ArrayList<Recognition> list)
    {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++)
        {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>()
                            {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs)
                                {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i)
            {
                if (list.get(i).getDetectedClass() == k)
                {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0)
            {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++)
                {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh)
                    {
                        pq.add(detection);
                    }
                }
            }
        }

        return nmsList;
    }

    protected float mNmsThresh = 0.25f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b)
    {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;

        return area;
    }

    protected float box_union(RectF a, RectF b)
    {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;

        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;

        return right - left;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int inputSize)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i)
        {
            for (int j = 0; j < inputSize; ++j)
            {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return byteBuffer;
    }

    private ArrayList<Recognition> getDetectionsForTiny(ByteBuffer byteBuffer, Bitmap bitmap)
    {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][OUTPUT_WIDTH_TINY[0]][4]);
        outputMap.put(1, new float[1][OUTPUT_WIDTH_TINY[1]][labels.size()]);
        Object[] inputArray = {byteBuffer};
        try
        {
            yoloIdentifier.runForMultipleInputsOutputs(inputArray, outputMap);
        }
        catch (Exception ex)
        {
            String msg = ex.getMessage();
        }


        int gridWidth = OUTPUT_WIDTH_TINY[0];
        float[][][] bboxes = (float [][][]) outputMap.get(0);
        float[][][] out_score = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth;i++)
        {
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); c++)
            {
                classes [c] = out_score[0][i][c];
            }

            for (int c = 0; c < labels.size(); ++c)
            {
                if (classes[c] > maxClass)
                {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float score = maxClass;
            if (score > getObjThresh() )
            {
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));

                detections.add(new Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }

        return detections;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap)
    {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap, INPUT_SIZE);

        final ArrayList<Recognition> detections = getDetectionsForTiny(byteBuffer, bitmap);

        final ArrayList<Recognition> onlyDogClassDetections = getOnlyDogClasses(detections);
        final ArrayList<Recognition> dogRecognitions = nms(onlyDogClassDetections);
        final ArrayList<Recognition> dogBreedResults = getDogBreedBoxes(dogRecognitions, bitmap);

        return dogBreedResults;
    }

    private ArrayList<Recognition> getDogBreedBoxes(ArrayList<Recognition> results, Bitmap bitmap)
    {
        ArrayList<Recognition> dogBreedBoxes = new ArrayList<>();
        for(final Recognition result : results)
        {
            RectF box = result.getLocation();
            int boxWidth = (int)(box.right - box.left);
            int boxHeight = (int)(box.bottom - box.top);
            Bitmap cropImage = Bitmap.createBitmap(bitmap, (int)box.left, (int)box.top, boxWidth, boxHeight);
            Bitmap resizedMobilenetBitmap = cropImage.createScaledBitmap(cropImage, 224, 224, false);

            ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedMobilenetBitmap, 224);
            float[][] output = new float[1][1001];
            try
            {
                mobileNetLite.run(byteBuffer, output);
                float[] class_scores = output[0];
                double minValue = DetectorActivity.MINIMUM_CONFIDENCE_SCORE;
                for(int jj = 0; jj < class_scores.length; jj++)
                {
                    if(class_scores[jj] > minValue)
                    {
                        String code =  indexToCode.get(new Integer(jj));
                        String breedName = codesToHuman.get(code);
                        Recognition resultWithBreed = result;
                        resultWithBreed.setBreedName(breedName);
                        dogBreedBoxes.add(resultWithBreed);
                    }
                }
            }
            catch (Exception ex)
            {
                LOGGER.d(ex.getMessage());
            }
        }

        return dogBreedBoxes;
    }

    private ArrayList<Recognition> getOnlyDogClasses(ArrayList<Recognition> results)
    {
        ArrayList<Recognition> onlyDogClasses = new ArrayList<>();
        for (Recognition item : results)
        {
            Integer resultClass = new Integer(item.getDetectedClass());
            if (possibleDetectionClasses.contains(resultClass))
            {
                onlyDogClasses.add(item);
            }
        }

        return onlyDogClasses;
    }
}
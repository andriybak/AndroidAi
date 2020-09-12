package com.dogbreed.detector;


import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker
{
    private static final float TEXT_SIZE_DIP = 18;
    private static final float MIN_SIZE = 16.0f;
    private static final float BOX_TOLERANCE_X = 50.0f;
    private static final float BOX_TOLERANCE_Y = 50.0f;

    private static final int[] COLORS =
    {
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.WHITE,
            Color.parseColor("#55FF55"),
            Color.parseColor("#FFA500"),
            Color.parseColor("#FF8888"),
            Color.parseColor("#AAAAFF"),
            Color.parseColor("#FFFFAA"),
            Color.parseColor("#55AAAA"),
            Color.parseColor("#AA33AA"),
            Color.parseColor("#0D0068")
    };

    private final Logger logger = new Logger();
    private final Queue<Integer> availableColors = new LinkedList<Integer>();
    private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
    private List<TrackedRecognition> oldTrackedObjects = new LinkedList<>();
    private final Paint boxPaint = new Paint();
    private final float textSizePx;
    private final BorderedText borderedText;
    private Matrix frameToCanvasMatrix;
    private int frameWidth;
    private int frameHeight;
    private int sensorOrientation;

    public MultiBoxTracker(final Context context)
    {
        for (final int color : COLORS)
        {
            availableColors.add(color);
        }

        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Style.STROKE);
        boxPaint.setStrokeWidth(10.0f);
        boxPaint.setStrokeCap(Cap.ROUND);
        boxPaint.setStrokeJoin(Join.ROUND);
        boxPaint.setStrokeMiter(100);

        textSizePx =TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
    }

    public synchronized void setFrameConfiguration(final int width, final int height, final int sensorOrientation)
    {
        frameWidth = width;
        frameHeight = height;
        this.sensorOrientation = sensorOrientation;
    }

    public synchronized void trackResults(final List<Classifier.Recognition> results, final long timestamp)
    {
        logger.i("Processing %d results from %d", results.size(), timestamp);
        processResults(results);
    }

    private Matrix getFrameToCanvasMatrix() {
        return frameToCanvasMatrix;
    }

    public synchronized void draw(final Canvas canvas)
    {
        final boolean rotated = sensorOrientation % 180 == 90;
        final float multiplier = Math.min(
                                canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                                canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
        frameToCanvasMatrix = ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                        (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                        sensorOrientation,
                        false);

        for (final TrackedRecognition recognition : trackedObjects)
        {
            final RectF trackedPos = new RectF(recognition.location);

            getFrameToCanvasMatrix().mapRect(trackedPos);
            boxPaint.setColor(recognition.color);

            float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

            final String labelString =
                    !TextUtils.isEmpty(recognition.title)
                            ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
                            : String.format("%.2f", (100 * recognition.detectionConfidence));

            float textPosX = trackedPos.left > 0.0 ? trackedPos.left : 0.0f;
            float textPosY = trackedPos.top > 0.0 ? trackedPos.top : 0.0f;
            if (textPosY > canvas.getHeight())
            {
                textPosY = trackedPos.bottom;
            }

            borderedText.drawText(canvas, textPosX + cornerSize, textPosY, labelString + "%", boxPaint);
        }
    }

    private void processResults(final List<Classifier.Recognition> results)
    {
        final List<Pair<Float, Classifier.Recognition>> rectsToTrack = new LinkedList<Pair<Float, Classifier.Recognition>>();
        for (final Classifier.Recognition result : results)
        {
            if (result.getLocation() == null)
            {
                continue;
            }

            final RectF detectionFrameRect = new RectF(result.getLocation());
            if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE)
            {
                logger.w("Degenerate rectangle! " + detectionFrameRect);
                continue;
            }

            rectsToTrack.add(new Pair<Float, Classifier.Recognition>(result.getConfidence(), result));
        }

        if(trackedObjects.size() > 0)
            oldTrackedObjects = new LinkedList<>(trackedObjects);

        trackedObjects.clear();

        if (rectsToTrack.isEmpty())
        {
            logger.v("Nothing to track, aborting.");
            return;
        }

        for (final Pair<Float, Classifier.Recognition> potential : rectsToTrack)
        {
            TrackedRecognition trackedRecognition = new TrackedRecognition();

            final int similarBoxIndex = GetSimilarBoxByLocation(potential.second, oldTrackedObjects);
            if (similarBoxIndex != -1)
            {
                TrackedRecognition oldRecognition = oldTrackedObjects.get(similarBoxIndex);
                trackedRecognition = UpdateExistingTrackedentry(oldRecognition, potential);
            }
            else
            {
                trackedRecognition = CreateNewTrackedEntry(potential);
                trackedRecognition.color = COLORS[trackedObjects.size()];
            }

            trackedObjects.add(trackedRecognition);

            if (trackedObjects.size() >= COLORS.length)
            {
                break;
            }
        }
    }

    private static TrackedRecognition UpdateExistingTrackedentry(TrackedRecognition existingEntry, final Pair<Float, Classifier.Recognition> recognition)
    {
        TrackedRecognition trackedRecognition = existingEntry;
        trackedRecognition.detectionConfidence = Math.max(recognition.first, existingEntry.detectionConfidence);

        if (recognition.second.getBreedName() != null)
        {
            trackedRecognition.title = recognition.second.getBreedName();
        }

        return trackedRecognition;
    }

    private static TrackedRecognition CreateNewTrackedEntry(final Pair<Float, Classifier.Recognition> recognition)
    {
        TrackedRecognition trackedRecognition = new TrackedRecognition();
        trackedRecognition.detectionConfidence = recognition.first;
        trackedRecognition.location = new RectF(recognition.second.getLocation());
        if (recognition.second.getBreedName() != null)
        {
            trackedRecognition.title = recognition.second.getBreedName();
        }
        else
        {
            trackedRecognition.title = recognition.second.getTitle();
        }

        return trackedRecognition;
    }

    private static int GetSimilarBoxByLocation(final Classifier.Recognition newRecognition, final List<TrackedRecognition> oldTrackedObjects)
    {
        final RectF newLocation = newRecognition.getLocation();
        final float centerX = newLocation.centerX();
        final float centerY = newLocation.centerY();

        for(int ii = 0; ii < oldTrackedObjects.size(); ii++)
        {
            final TrackedRecognition oldObject = oldTrackedObjects.get(ii);
            final float oldCenterX = oldObject.location.centerX();
            final float oldCenterY = oldObject.location.centerY();
            final float distanceX = Math.abs(centerX - oldCenterX);
            final float distanceY = Math.abs(centerY - oldCenterY);
            if (distanceX < BOX_TOLERANCE_X && distanceY < BOX_TOLERANCE_Y)
            {
                return ii;
            }
        }

        return -1;
    }

    private static class TrackedRecognition {
        RectF location;
        float detectionConfidence;
        int color;
        String title;
    }
}

package com.camera.cameratest;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.view.View;

import java.util.List;


@SuppressLint("AppCompatCustomView")
class DrawView extends View {

    //private Drawable mCustomImage;
    List<Classifier.Recognition> m_results;
    List<String> m_breedNames;

    public DrawView(Context context, List<Classifier.Recognition> results, List<String> breedNames) {
        super(context);
        m_results = results;
        m_breedNames = breedNames;
    }

    public void SetBoxes(RectF box)
    {
        m_results.get(0).setLocation(box);
    }

    @Override
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if(m_results.size() > 0)
        {
            Paint paint = new Paint();
            paint.setColor(Color.BLUE);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(10);

            for(int ii=0;ii<m_results.size() && ii < m_breedNames.size();ii++)
            {
                Classifier.Recognition result = m_results.get(ii);
                String breeds = m_breedNames.get(ii).split(",")[0];
                RectF box = result.getLocation();
                canvas.drawRect(box, paint);
                float width = paint.measureText(breeds);
                paint.setColor(Color.BLACK);
                paint.setColor(Color.WHITE);
                paint.setTextSize(70);
                paint.setStrokeWidth(3);
                canvas.drawText(breeds,box.left, box.top, paint);
            }
        }
    }
}

package com.camera.cameratest;

import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class ClassesParser {

    HashMap<Integer, String> indexToCode;
    HashMap<String, String> codesToHuman;

    public ClassesParser(AssetManager assets)
    {
        indexToCode = new HashMap<Integer, String>();
        codesToHuman = new HashMap<String, String>();
        InitializeMap(assets);
    }

    public String GetValueByIndex(Integer index)
    {
        String code =  indexToCode.get(index);
        return codesToHuman.get(code);
    }

    private void InitializeMap(AssetManager assets)
    {
        this.ReadCodesFile(assets);
        this.FillCodesWithValues(assets);
    }

    private void FillCodesWithValues(AssetManager assets)
    {
        BufferedReader reader = null;
        try
        {
            reader = new BufferedReader(new InputStreamReader(assets.open("code_human.txt")));
            String mLine;
            while ((mLine = reader.readLine()) != null)
            {
                String[] codeAndName = mLine.trim().split("\t");
                if(codeAndName.length >= 2)
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

    private void ReadCodesFile(AssetManager assets)
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
}

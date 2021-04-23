package com.example.objectdetectordemo;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.renderscript.ScriptGroup;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.BitmapFactory;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button buttonDetect = (Button) findViewById(R.id.detect);

        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O){
            requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE},1);
        }
        buttonLoadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");

                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i,RESULT_LOAD_IMAGE);
            }
        });

        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap = null;
                Module module = null;

                //Get the image from imageView
                ImageView imageView = (ImageView) findViewById(R.id.image);

                try {
                    //read the image as bitmap
                    bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

                    // Reshape the image to 400x400
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

                    //laod the model file
                    module = Module.load(fetchModelFile(MainActivity.this, "resnet18_traced.pt"));
                } catch (IOException e) {
                    finish();
                }

                //Input Tensor
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );

                //model forward pass to run on the input image
                assert module != null;
                final Tensor output = module.forward(IValue.from(input)).toTensor();

                // get the output scores
                final float[] score_arr = output.getDataAsFloatArray();

                //Get the index of the max value of the array, which is the predicted probability of the class
                float max_score = -Float.MAX_VALUE;
                int max_index = -1;

                for(int i = 0; i<score_arr.length; i++){
                    if(max_score<score_arr[i]){
                        max_score = score_arr[i];
                        max_index = i;
                    }
                }

                // Get the name of the class from the ModelClasses
                String detectedClass = ModelClasses.MODEL_CLASSES[max_index];
                String score = String.valueOf(max_score);

                //Write the detectedClass into the result view
                TextView textView = findViewById(R.id.result_text);
                textView.setText(detectedClass);
                TextView textView1 = findViewById(R.id.result_score);
                textView1.setText(score);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode,resultCode,data);

        if(requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK){
            Uri selectedImage = data.getData();

            String[] filePathColumn  = {MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(selectedImage, filePathColumn,null,null,null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView =(ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);

        }

    }

    public static String fetchModelFile(Context context, String modelName) throws IOException{
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length()>0){
            return file.getAbsolutePath();
        }

        try(InputStream is = context.getAssets().open(modelName)){
            try(OutputStream os = new FileOutputStream(file)){
                byte[] buffer = new byte[4*1024];
                int read;
                while((read = is.read(buffer))!=-1){
                    os.write(buffer,0,read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
package com.example.objectdetectionwithtflite

import android.app.Dialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.SurfaceView
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector

private lateinit var CameraBridgeViewBase : CameraBridgeViewBase
private lateinit var toggle : ImageView
private lateinit var model : ObjectDetector

class ObjectDetection : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.object_detection)

        createObjectDetector()
        checkPermission()

        CameraBridgeViewBase = findViewById(R.id.CameraView)
        toggle = findViewById(R.id.toggle_indicator2)

        CameraBridgeViewBase.visibility = SurfaceView.VISIBLE
        CameraBridgeViewBase.setCvCameraViewListener(this)
        CameraBridgeViewBase.setCameraPermissionGranted()
        CameraBridgeViewBase.enableView()

        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV Loaded Successfully", Toast.LENGTH_LONG)
        } else {
            Toast.makeText(this, "OpenCV Loading Unsuccessful", Toast.LENGTH_LONG)
        }

        toggle.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat? {
        val frame: Mat = inputFrame?.rgba() ?: return null
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB)
        val bm = convMatToBM(frame)

        return startDetection(model, bm)
    }

    private fun checkPermission(){
        if (applicationContext.packageManager.hasSystemFeature(
                PackageManager.FEATURE_CAMERA)){
            Toast.makeText(this, "Camera Permissions Granted", Toast.LENGTH_LONG)
        } else {
            Toast.makeText(this, "Please Grant Camera Permissions", Toast.LENGTH_LONG)
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}

    override fun onCameraViewStopped() {}

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV Loading: Unsuccessful", Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(this, "OpenCV Loading Successful", Toast.LENGTH_LONG).show()
        }
        CameraBridgeViewBase.enableView()
    }

    override fun onPause() {
        super.onPause()
        if (CameraBridgeViewBase != null){
            CameraBridgeViewBase.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (CameraBridgeViewBase != null){
            CameraBridgeViewBase.disableView()
        }
    }

    private fun convMatToBM(mat: Mat): Bitmap {
        val bm: Bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bm)
        return bm
    }

    private fun createObjectDetector() {
        // Specify options
        val options = org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(intent.getIntExtra("MAX_RESULTS", 5))
            .setScoreThreshold(intent.getFloatExtra("SCORE_THRESH", 0.5F))
            .build()

        // Create object detector by passing in pre-trained model
        model = ObjectDetector.createFromFileAndOptions(this, "model.tflite", options)
    }

    private fun startDetection(model : ObjectDetector, bitmap: Bitmap) : Mat {
        // Convert bitmap to tensor
        val tensorImage = TensorImage.fromBitmap(bitmap)

        // Get detection results
        val results = model.detect(tensorImage)
        val result = results.map {
            val category = it.categories.first()

            // Get predicted label and confidence score
            val label = category.label
            val score = category.score * 100
            BoundingBoxData(it.boundingBox, label, score.toInt())
        }

        // Draw bounding boxes on uploaded image and return result
        val bbBM =  drawBoundingBox(bitmap, result)

        val imgMat = Mat(bbBM.height, bbBM.width, CvType.CV_8UC1)
        Utils.bitmapToMat(bbBM, imgMat)
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_BGR2RGB)

        return imgMat
    }

    fun drawBoundingBox(bitmap: Bitmap, results:List<BoundingBoxData>) : Bitmap {
        // Copy bitmap
        val copyBM = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Instantiate canvas and pen for drawing
        val canvas = Canvas(copyBM)
        val pen = Paint()

        // Iterate through results list
        for (result in results){
            pen.style = Paint.Style.STROKE
            if (result.score > 50) pen.color = Color.GREEN else pen.color = Color.RED
            pen.strokeWidth = 5f

            // Draw bounding box
            val bbox = result.BoundingBox
            canvas.drawRect(bbox, pen)

            val tagSize = Rect(0, 0, 0, 0)

            // Draw text
            val text = "${result.label} | ${result.score}"
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.strokeWidth = 5F
            pen.color = Color.YELLOW
            pen.textSize = 75F

            pen.getTextBounds(text, 0, text.length, tagSize)
            val fontSize: Float = pen.textSize * bbox.width() / tagSize.width()

            if (fontSize < pen.textSize) pen.textSize = fontSize
            var margin = (bbox.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F

            canvas.drawText(text, bbox.left + margin,
                bbox.top + tagSize.height().times(1F), pen)
        }
        return copyBM
    }

    private fun setValues(score : Int, thresh : Float){

    }

    private fun displayModal() {
        // Create dialog
        val dialog = Dialog(this)
        dialog.setCancelable(false)

        // Set content view of dialog to Modal layout
        dialog.setContentView(R.layout.modal)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT)) // Set transparent bg

        val upload = dialog.findViewById<Button>(R.id.upload)
        val rlObj = dialog.findViewById<Button>(R.id.rlObj)
        val maxResultsEdit = dialog.findViewById<EditText>(R.id.MaxResults)
        val scoreThreshEdit = dialog.findViewById<EditText>(R.id.ScoreThreshold)

        upload.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        rlObj.setOnClickListener {
            dialog.dismiss()
        }
        dialog.show()
    }

    data class BoundingBoxData(val BoundingBox : RectF, val label : String, val score : Int)
}

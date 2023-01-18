package com.example.objectdetectionwithtflite

import android.app.Activity
import android.app.Dialog
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.animation.AnimationUtils
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.support.image.TensorImage
import java.io.IOException

private lateinit var toggle: ImageView
private lateinit var uploadBtn: Button
private lateinit var imageDisplay: ImageView

class MainActivity : AppCompatActivity() {

    var maxResults : Int = 5
    var scoreThresh : Float = 0.5F

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        displayModal()

        // Main image display
        imageDisplay = findViewById(R.id.image_display)

        toggle = findViewById(R.id.toggle_indicator) // Toggle modal
        uploadBtn = findViewById(R.id.uploadbtn) // Upload Button

        toggle.setOnClickListener {
            displayModal()
        }

        uploadBtn.setOnClickListener {
            getGalleryImage()
        }
    }

    private fun getGalleryImage() {
        val intent = Intent()
        intent.action = Intent.ACTION_GET_CONTENT
        intent.type = "image/*"
        resultLauncher.launch(intent)
    }

    private var resultLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data?.data
                try {
                    val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, data)
                    imageDisplay.setImageBitmap(createObjectDetector(bitmap))

                } catch (e: IOException) {
                    e.printStackTrace()
                    Toast.makeText(this, "Please Enable File Permissions", Toast.LENGTH_LONG)
                }
            }
        }

    private fun displayModal() {
        // Get animations from resource
        val popupAnim = AnimationUtils.loadAnimation(this, R.anim.modal_pop_up)
        val dropdownAnim = AnimationUtils.loadAnimation(this, R.anim.modal_pop_down)

        // Create dialog
        val dialog = Dialog(this)
        dialog.setCancelable(false)

        // Set content view of dialog to Modal layout
        dialog.setContentView(R.layout.modal)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT)) // Set transparent bg

        // Assign animations
        val toggle = findViewById<ImageView>(R.id.toggle_indicator)
        toggle.startAnimation(popupAnim)

        val modal = dialog.findViewById<ImageView>(R.id.modal_body)
        modal.startAnimation(popupAnim)

        val tfIcon = dialog.findViewById<ImageView>(R.id.tf_icon)
        tfIcon.startAnimation(popupAnim)

        val choose = dialog.findViewById<TextView>(R.id.choose)
        choose.startAnimation(popupAnim)

        val or = dialog.findViewById<TextView>(R.id.or)
        or.startAnimation(popupAnim)

        val upload = dialog.findViewById<Button>(R.id.upload)
        upload.startAnimation(popupAnim)

        val rlObj = dialog.findViewById<Button>(R.id.rlObj)
        rlObj.startAnimation(popupAnim)

        val maxResultsEdit = dialog.findViewById<EditText>(R.id.MaxResults)
        maxResultsEdit.startAnimation(popupAnim)

        val scoreThreshEdit = dialog.findViewById<EditText>(R.id.ScoreThreshold)
        scoreThreshEdit.startAnimation(popupAnim)

        upload.setOnClickListener {

            maxResults = if(maxResultsEdit.text.isNotEmpty()) maxResultsEdit.text.toString().toInt() else maxResults
            scoreThresh = if(scoreThreshEdit.text.isNotEmpty()) scoreThreshEdit.text.toString().toFloat() * 0.01F else scoreThresh

            // Closing animations
            modal.startAnimation(dropdownAnim)
            tfIcon.startAnimation(dropdownAnim)
            choose.startAnimation(dropdownAnim)
            or.startAnimation(dropdownAnim)
            upload.startAnimation(dropdownAnim)
            rlObj.startAnimation(dropdownAnim)
            maxResultsEdit.startAnimation(dropdownAnim)
            scoreThreshEdit.startAnimation(dropdownAnim)

            toggle.startAnimation(dropdownAnim)

            Handler(Looper.getMainLooper()).postDelayed({ dialog.dismiss() }, 500)
        }

        rlObj.setOnClickListener {

            maxResults = maxResultsEdit.text.toString().toInt()
            scoreThresh = scoreThreshEdit.text.toString().toFloat() * 0.01F

            val intent = Intent(this, ObjectDetection::class.java).also{
                it.putExtra("MAX_RESULTS", maxResultsEdit.text.toString().toInt())
                it.putExtra("SCORE_THRESH", scoreThreshEdit.text.toString().toFloat() * 0.01F)
            }
            startActivity(intent)
        }
        dialog.show()
    }

    private fun createObjectDetector(bitmap: Bitmap): Bitmap {
        // Convert bitmap to tensors
        val tensorImg = TensorImage.fromBitmap(bitmap)

        // Specify options
        val options = org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(maxResults)
            .setScoreThreshold(scoreThresh)
            .build()

        // Create object detector by passing in pre-trained model
        val model = org.tensorflow.lite.task.vision.detector.ObjectDetector.createFromFileAndOptions(this, "model.tflite", options)

        // Get detection results
        val results = model.detect(tensorImg)
        val result = results.map {
            val category = it.categories.first()

            // Get predicted label and confidence score
            val label = category.label
            val score = category.score * 100
            BoundingBoxData(it.boundingBox, label, score.toInt())
        }

        // Draw bounding boxes on uploaded image and return result
        return drawBoundingBox(bitmap, result)
    }

    private fun drawBoundingBox(bitmap: Bitmap, results:List<BoundingBoxData>) : Bitmap {
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

    data class BoundingBoxData(val BoundingBox : RectF, val label : String, val score : Int)
}


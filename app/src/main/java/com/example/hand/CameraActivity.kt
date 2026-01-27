package com.example.hand

import android.Manifest
import android.os.Bundle
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private val kLang = "th"
    private lateinit var i18n: I18n

    // ✅ กล้องหลัง
    private val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private val mirrorX = false

    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var btnToggle: FrameLayout
    private lateinit var imgToggle: ImageView
    private lateinit var btnBack: ImageButton

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var isRunning = false

    // ML
    private var runner: HolisticLandmarkerRunner? = null
    private var classifier: LitertSequenceClassifier? = null
    private var labelMap: LabelMap? = null
    private var thr: ThresholdConfig = ThresholdConfig()
    private var agg: PredictionAggregator? = null

    private val seq = SequenceBuffer(seqLen = ModelConfig.SEQ_T, featDim = ModelConfig.FEAT_F)

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else setStatus(i18nSafe("camera_permission_denied", "ไม่ได้รับสิทธิ์กล้อง"))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_camera)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)

        i18n = I18n.fromAssets(this, kLang)

        btnBack.setOnClickListener {
            stopRun(showSummary = false)
            finish()
        }

        // ✅ check assets ตาม ModelConfig
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists(ModelConfig.MODEL_THRESH_ASSET)) add(ModelConfig.MODEL_THRESH_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatus("❌ assets หาย:\n${missing.joinToString(", ")}")
            return
        }

        // init model
        try {
            labelMap = LabelMap.fromAssets(this, ModelConfig.MODEL_LABELS_ASSET)
            thr = ThresholdConfig.fromAssets(this, ModelConfig.MODEL_THRESH_ASSET)
            agg = PredictionAggregator(numClasses = labelMap!!.size)

            classifier = LitertSequenceClassifier(
                context = this,
                modelAssetName = ModelConfig.MODEL_TFLITE_ASSET,
                numClasses = labelMap!!.size,
                numThreads = 4
            )
        } catch (t: Throwable) {
            setStatus("❌ init fail: ${t.message}")
            return
        }

        // init holistic
        try {
            runner = HolisticLandmarkerRunner(
                context = this,
                modelAssetName = ModelConfig.HOLISTIC_TASK,
                mirrorX = mirrorX,
                onResult = { r -> onHolisticResult(r) },
                onError = { msg -> setStatus("❌ Holistic error: $msg") }
            )
        } catch (t: Throwable) {
            setStatus("❌ Holistic init fail: ${t.message}")
            return
        }

        btnToggle.setOnClickListener {
            if (!isRunning) startRun() else stopRun(showSummary = true)
        }

        setStatus(i18nSafe("ready_press_play", "✅ พร้อม (กดปุ่มเล่น)"))
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        agg?.reset()
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)

        // ✅ ระหว่างรันไม่โชว์คำแปล
        setStatus(i18nSafe("detecting", "กำลังจับท่า..."))
    }

    // ✅ stopRun() แบบเต็ม (ใช้ MIN/MAX)
    private fun stopRun(showSummary: Boolean) {
        isRunning = false
        imgToggle.setImageResource(android.R.drawable.ic_media_play)

        if (!showSummary) {
            setStatus(i18nSafe("stopped", "หยุดแล้ว"))
            return
        }

        val lm = labelMap
        val a = agg
        if (lm == null || a == null) {
            setStatus(i18nSafe("stopped", "หยุดแล้ว"))
            return
        }

        val best = a.bestResult(
            minCount = ModelConfig.MIN_COUNT_TO_ACCEPT,
            maxCount = ModelConfig.MAX_COUNT_TO_ACCEPT,
            minSum = ModelConfig.MIN_SUM_TO_ACCEPT,
            maxSum = ModelConfig.MAX_SUM_TO_ACCEPT,
            minAvg = ModelConfig.MIN_AVG_SCORE_TO_ACCEPT,
            maxAvg = ModelConfig.MAX_AVG_SCORE_TO_ACCEPT
        )

        if (best == null) {
            setStatus(i18nSafe("not_sure", "ยังไม่มั่นใจ ลองใหม่"))
            return
        }

        val key = lm[best.idx]
        val thai = i18n.t(key)
        setStatus("${i18nSafe("summary", "ผลสรุป")}: $thai")
    }

    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        if (!isRunning) return

        // Gate: ไม่มีคน -> ไม่ทำนับ
        if (!hasPerson(result)) {
            seq.reset()
            agg?.reset()
            setStatus(i18nSafe("no_person", "ไม่พบคน"))
            return
        }

        val frame = Holistic258Extractor.extract(result)
        seq.add(frame)
        if (!seq.isFull()) return

        val probs = classifier?.predict(seq.toFlatFloatArray()) ?: return
        val top = top1top2(probs)

        // per-step threshold: tau/delta
        val pass = top.topScore >= thr.tau && (top.topScore - top.secondScore) >= thr.delta
        if (pass) agg?.add(top.topIdx, top.topScore)

        // ✅ ไม่โชว์คำตอบระหว่างรัน (โชว์ตอน stopRun เท่านั้น)
    }

    private fun hasPerson(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.isEmpty()) return false

        var sumVis = 0f
        var good = 0
        for (lm in pose) {
            val vis = try { lm.visibility().orElse(0f) } catch (_: Throwable) { 0f }
            sumVis += vis
            if (vis >= 0.5f) good++
        }
        val meanVis = sumVis / pose.size
        return meanVis >= 0.25f && good >= 8
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            val r = runner
            if (r != null) {
                analysis.setAnalyzer(
                    cameraExecutor,
                    HolisticFrameAnalyzer(
                        enabled = { isRunning },
                        runner = r
                    )
                )
            }

            provider.unbindAll()
            provider.bindToLifecycle(this, cameraSelector, preview, analysis)

            setStatus(i18nSafe("ready_press_play", "✅ พร้อม (กดปุ่มเล่น)"))
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setStatus(msg: String) {
        runOnUiThread { if (::txtResult.isInitialized) txtResult.text = msg }
    }

    private fun i18nSafe(key: String, fallback: String): String {
        if (!::i18n.isInitialized) return fallback
        val v = i18n.t(key)
        return if (v == key) fallback else v
    }

    private fun assetExists(name: String): Boolean {
        return try {
            assets.open(name).close()
            true
        } catch (_: Throwable) {
            false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}
        cameraExecutor.shutdown()
    }

    override fun onBackPressed() {
        stopRun(showSummary = false)
        super.onBackPressed()
    }

    // ===== Top2 helper (ไม่ต้องพึ่ง ArgMax.kt) =====
    private data class Top2(
        val topIdx: Int,
        val topScore: Float,
        val secondIdx: Int,
        val secondScore: Float
    )

    private fun top1top2(probs: FloatArray): Top2 {
        var topIdx = -1
        var secondIdx = -1
        var topScore = Float.NEGATIVE_INFINITY
        var secondScore = Float.NEGATIVE_INFINITY

        for (i in probs.indices) {
            val v = probs[i]
            if (v > topScore) {
                secondScore = topScore
                secondIdx = topIdx
                topScore = v
                topIdx = i
            } else if (v > secondScore) {
                secondScore = v
                secondIdx = i
            }
        }
        if (topIdx < 0) topIdx = 0
        if (secondIdx < 0) secondIdx = topIdx
        return Top2(topIdx, topScore, secondIdx, secondScore)
    }
}

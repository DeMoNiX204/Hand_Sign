package com.example.hand

import android.Manifest
import android.os.Bundle
import android.util.Size
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

    // ===== Camera config =====
    private val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    private var cameraProvider: ProcessCameraProvider? = null
    // ใช้ Executor
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // ===== UI =====
    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var btnToggle: FrameLayout
    private lateinit var imgToggle: ImageView
    private lateinit var btnBack: ImageButton

    // ===== State =====
    private var isRunning = false
    private var lastUiState: String = ""

    // ===== ML =====
    private var runner: HolisticLandmarkerRunner? = null
    private var classifier: LitertSequenceClassifier? = null
    private var labelMap: LabelMap? = null
    private var thr: ThresholdConfig = ThresholdConfig()
    private var agg: PredictionAggregator? = null

    private val seq = SequenceBuffer(seqLen = ModelConfig.SEQ_T, featDim = ModelConfig.FEAT_F)

    // EMA Smoothing
    private var prevProbs: FloatArray? = null
    private val alphaEMA = 0.2f

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isAllowed ->
        if (isAllowed) startCamera()
        else setStatusState("ไม่ได้รับอนุญาตให้ใช้กล้อง")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_camera)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        // UI 
        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)

        btnBack.setOnClickListener {
            stopRun(showSummary = false)
            finish()
        }

        // Check Assets
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatusState("assets หาย: ${missing.joinToString(", ")}")
            return
        }

        // Load i18n
        try {
            i18n = I18n.fromAssets(this, kLang)
        } catch (e: Exception) {
            setStatusState("i18n error: ${e.message}")
            return
        }

        // Init ML
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

            runner = HolisticLandmarkerRunner(
                context = this,
                modelAssetName = ModelConfig.HOLISTIC_TASK,
                onResult = { r -> onHolisticResult(r) },
                onError = { msg -> setStatusState("ML Error: $msg") }
            )
        } catch (t: Throwable) {
            setStatusState("Init Fail: ${t.message}")
            return
        }

        // Ready
        btnToggle.setOnClickListener {
            if (!isRunning) startRun() else stopRun(showSummary = true)
        }

        setStatusState("พร้อม (กดปุ่มเล่น)")
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        Holistic258Extractor.reset()
        agg?.reset()
        prevProbs = null // รีเซ็ต EMA
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)
        setStatusState("กำลังจับท่า...")
    }

    private fun stopRun(showSummary: Boolean) {
        isRunning = false
        imgToggle.setImageResource(android.R.drawable.ic_media_play)

        if (!showSummary) {
            setStatusState("หยุดแล้ว")
            return
        }

        val lm = labelMap ?: return
        val a = agg ?: return
        val noActionIdx = lm.indexOf("no_action")

        val best = a.bestResultExclude(
            excludeIdx = noActionIdx,
            minCount = ModelConfig.MIN_COUNT_TO_ACCEPT,
            minSum = ModelConfig.MIN_SUM_TO_ACCEPT,
            minAvg = ModelConfig.MIN_AVG_SCORE_TO_ACCEPT
        )

        if (best == null) {
            setStatusState("ยังไม่มั่นใจ ลองใหม่")
        } else {
            val key = lm[best.idx]
            val percent = (best.avg * 100).toInt()
            setStatusState("${"ผลลัพธ์"}: ${i18n.t(key)} ($percent%)")
        }
    }

    // ===== Core Logic =====
    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        if (!isRunning) return
        if (labelMap == null || classifier == null) return

        if (!hasPerson(result)) {
            setStatusState("ไม่พบคน")
            return
        }
        if (!hasBothArms(result)) {
            setStatusState("เห็นแขนไม่ครบ")
            return
        }

        setStatusState("กำลังจับท่า...")

        val frame = Holistic258Extractor.extract(result)
        seq.add(frame)
        if (!seq.isFull()) return

        // Raw Probabilities
        val rawProbs = classifier?.predict(seq.toFlatFloatArray()) ?: return

        //  คำนวณ EMA Smoothing
        val probs = if (prevProbs == null) {
            rawProbs
        } else {
            FloatArray(rawProbs.size) { i ->
                // สูตร: smoothed = alpha * new + (1-alpha) * old
                alphaEMA * rawProbs[i] + (1f - alphaEMA) * prevProbs!![i]
            }
        }
        prevProbs = probs // จำค่าไว้ใช้รอบหน้า

        //  ใช้ค่า probs ที่ Smooth แล้วไปคำนวณต่อ
        val top = getTopResult(probs)
        val key = labelMap!![top.topIdx]

        val rule = thr.forLabel(key)

        val pass = (top.topScore >= rule.tau)

        if (pass && key != "no_action") {
            agg?.add(top.topIdx, top.topScore)
        }
    }

    // ... (ฟังก์ชัน Helper hasPerson, hasBothArms, top1top2 เหมือนเดิม) ...
    private fun hasPerson(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.isEmpty()) return false
        var good = 0
        for (lm in pose) if ((lm.visibility().orElse(0f)) >= 0.5f) good++
        return good >= 8
    }

    private fun hasBothArms(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.size < 17) return false
        fun ok(i: Int): Boolean {
            val lm = pose[i]
            return (lm.x() in 0f..1f && lm.y() in 0f..1f && (lm.visibility().orElse(0f) >= 0.35f))
        }
        return (ok(11) && ok(13) && ok(15)) && (ok(12) && ok(14) && ok(16))
    }

    private fun getTopResult(probs: FloatArray): TopResult {
        var topIdx = 0; var topScore = -1f
        var secIdx = 0; var secScore = -1f
        for (i in probs.indices) {
            val v = probs[i]
            if (v > topScore) { secScore = topScore; secIdx = topIdx; topScore = v; topIdx = i }
            else if (v > secScore) { secScore = v; secIdx = i }
        }
        return TopResult(topIdx, topScore)
    }

    // ===== Camera =====
    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))

    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(720, 1280))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val r = runner
        if (r != null) {
            analysis.setAnalyzer(cameraExecutor, HolisticFrameAnalyzer({ true }, r))
        }

        provider.unbindAll()
        provider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    private fun setStatusState(msg: String) {
        if (msg == lastUiState) return
        lastUiState = msg
        if (::txtResult.isInitialized) runOnUiThread { txtResult.text = msg }
    }


    private fun assetExists(n: String) = try { assets.open(n).close(); true } catch (_: Throwable) { false }

    //  onDestroy เพื่อป้องกัน Crash ตอนกด Back
    override fun onDestroy() {
        // หยุดรับภาพจากกล้องทันที
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}

        // ปิด Thread ที่ประมวลผล
        cameraExecutor.shutdown()

        // ปิด Model
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}

        super.onDestroy()
    }

    private data class TopResult(val topIdx: Int, val topScore: Float)
}
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

    // 🔥 1. ตั้งค่า mirrorX = true เพื่อจำลองการ flip แบบ Python
    private val mirrorX = true

    private var cameraProvider: ProcessCameraProvider? = null
    // ใช้ Executor แบบนี้ปลอดภัยกว่า
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

    // 🔥 2. เพิ่มตัวแปรสำหรับ EMA Smoothing
    private var prevProbs: FloatArray? = null
    private val alphaEMA = 0.2f // ค่า Alpha 0.2 (ตรงกับ Python)

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
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

        // 1. Init UI ก่อนเสมอ
        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)

        btnBack.setOnClickListener {
            // หยุดการทำงานก่อนปิด
            stopRun(showSummary = false)
            finish()
        }

        // 2. Check Assets (กันเข้าแล้วเด้ง)
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatusState("❌ assets หาย: ${missing.joinToString(", ")}")
            return
        }

        // 3. Load i18n
        try {
            i18n = I18n.fromAssets(this, kLang)
        } catch (e: Exception) {
            setStatusState("❌ i18n error: ${e.message}")
            return
        }

        // 4. Init ML
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
                mirrorX = mirrorX,
                onResult = { r -> onHolisticResult(r) },
                onError = { msg -> setStatusState("❌ ML Error: $msg") }
            )
        } catch (t: Throwable) {
            setStatusState("❌ Init Fail: ${t.message}")
            return
        }

        // Ready
        btnToggle.setOnClickListener {
            if (!isRunning) startRun() else stopRun(showSummary = true)
        }

        setStatusState(i18nSafe("ready_press_play", "✅ พร้อม (กดปุ่มเล่น)"))
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        agg?.reset()
        prevProbs = null // รีเซ็ต EMA เมื่อเริ่มใหม่
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)
        setStatusState(i18nSafe("detecting", "กำลังจับท่า..."))
    }

    private fun stopRun(showSummary: Boolean) {
        isRunning = false
        imgToggle.setImageResource(android.R.drawable.ic_media_play)

        if (!showSummary) {
            setStatusState(i18nSafe("stopped", "หยุดแล้ว"))
            return
        }

        val lm = labelMap ?: return
        val a = agg ?: return
        val noActionIdx = findLabelIndex(lm, "no_action")

        val best = a.bestResultExclude(
            excludeIdx = noActionIdx,
            minCount = ModelConfig.MIN_COUNT_TO_ACCEPT,
            maxCount = ModelConfig.MAX_COUNT_TO_ACCEPT,
            minSum = ModelConfig.MIN_SUM_TO_ACCEPT,
            maxSum = ModelConfig.MAX_SUM_TO_ACCEPT,
            minAvg = ModelConfig.MIN_AVG_SCORE_TO_ACCEPT,
            maxAvg = ModelConfig.MAX_AVG_SCORE_TO_ACCEPT
        )

        if (best == null) {
            setStatusState(i18nSafe("not_sure", "ยังไม่มั่นใจ ลองใหม่"))
        } else {
            val key = lm[best.idx]
            setStatusState("${i18nSafe("summary", "ผลลัพธ์")}: ${i18n.t(key)}")
        }
    }

    // ===== Core Logic =====
    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        if (!isRunning) return
        if (labelMap == null || classifier == null) return

        if (!hasPerson(result)) {
            setStatusState(i18nSafe("no_person", "ไม่พบคน"))
            return
        }
        if (!hasBothArms(result)) {
            setStatusState(i18nSafe("need_both_arms", "เห็นแขนไม่ครบ"))
            return
        }

        setStatusState(i18nSafe("detecting", "กำลังจับท่า..."))

        val frame = Holistic258Extractor.extract(result)
        seq.add(frame)
        if (!seq.isFull()) return

        // 1. ได้ค่าดิบ (Raw Probabilities)
        val rawProbs = classifier?.predict(seq.toFlatFloatArray()) ?: return

        // 🔥 3. คำนวณ EMA Smoothing (เพื่อให้ค่านิ่งเท่า Python)
        val probs = if (prevProbs == null) {
            rawProbs
        } else {
            FloatArray(rawProbs.size) { i ->
                // สูตร: smoothed = alpha * new + (1-alpha) * old
                alphaEMA * rawProbs[i] + (1f - alphaEMA) * prevProbs!![i]
            }
        }
        prevProbs = probs // จำค่าไว้ใช้รอบหน้า

        // 4. ใช้ค่า probs ที่ Smooth แล้วไปคำนวณต่อ
        val top = top1top2(probs)
        val key = labelMap!![top.topIdx]

        val rule = thr.forLabel(key)
        val diff = top.topScore - top.secondScore
        val pass = (top.topScore >= rule.tau) && (diff >= rule.delta)

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

    private fun top1top2(probs: FloatArray): Top2 {
        var topIdx = 0; var topScore = -1f
        var secIdx = 0; var secScore = -1f
        for (i in probs.indices) {
            val v = probs[i]
            if (v > topScore) { secScore = topScore; secIdx = topIdx; topScore = v; topIdx = i }
            else if (v > secScore) { secScore = v; secIdx = i }
        }
        return Top2(topIdx, topScore, secIdx, secScore)
    }

    private fun findLabelIndex(lm: LabelMap, t: String): Int? {
        for (i in 0 until lm.size) if (lm[i] == t) return i
        return null
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
        try { Holistic258Extractor.setMirror(mirrorX) } catch (_: Throwable) {}

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

    private fun i18nSafe(key: String, fallback: String): String {
        return if (::i18n.isInitialized) i18n.t(key) else fallback
    }

    private fun assetExists(n: String) = try { assets.open(n).close(); true } catch (_: Throwable) { false }

    // 🔥🔥 จุดสำคัญ: แก้ onDestroy เพื่อป้องกัน Crash ตอนกด Back 🔥🔥
    override fun onDestroy() {
        // 1. หยุดรับภาพจากกล้องทันที (สำคัญมาก)
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}

        // 2. ปิด Thread ที่ประมวลผล
        cameraExecutor.shutdown()

        // 3. ปิด Model
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}

        super.onDestroy()
    }

    private data class Top2(val topIdx: Int, val topScore: Float, val secondIdx: Int, val secondScore: Float)
}
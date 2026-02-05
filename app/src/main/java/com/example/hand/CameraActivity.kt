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
import java.util.Locale
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private val kLang = "th"

    private lateinit var i18n: I18n

    // ===== Camera config (กล้องธรรมดาใช้กล้องหลัง) =====
    private val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private val mirrorX = false

    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // ===== UI =====
    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var btnToggle: FrameLayout
    private lateinit var imgToggle: ImageView
    private lateinit var btnBack: ImageButton

    // ===== State =====
    private var isRunning = false

    // กัน spam setStatus ทุกเฟรม
    private var lastUiState: String = ""

    // ===== ML =====
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
        else setStatusState(i18nSafe("camera_permission_denied", "ไม่ได้รับสิทธิ์กล้อง"))
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

        // ---- assets check ----
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists(ModelConfig.MODEL_THRESH_ASSET)) add(ModelConfig.MODEL_THRESH_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatusState("❌ assets หาย: ${missing.joinToString(", ")}")
            return
        }

        // ---- init models ----
        try {
            labelMap = LabelMap.fromAssets(this, ModelConfig.MODEL_LABELS_ASSET)
            thr = ThresholdConfig.fromAssets(this, ModelConfig.MODEL_THRESH_ASSET) // ✅ per-class
            agg = PredictionAggregator(numClasses = labelMap!!.size)

            classifier = LitertSequenceClassifier(
                context = this,
                modelAssetName = ModelConfig.MODEL_TFLITE_ASSET,
                numClasses = labelMap!!.size,
                numThreads = 4
            )
        } catch (t: Throwable) {
            setStatusState("❌ init fail: ${t.message}")
            return
        }

        // ---- init holistic runner ----
        try {
            runner = HolisticLandmarkerRunner(
                context = this,
                modelAssetName = ModelConfig.HOLISTIC_TASK,
                mirrorX = mirrorX,
                onResult = { r -> onHolisticResult(r) },
                onError = { msg -> setStatusState("❌ Holistic error: $msg") }
            )
        } catch (t: Throwable) {
            setStatusState("❌ Holistic init fail: ${t.message}")
            return
        }

        // toggle start/stop
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
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)

        // ✅ ระหว่างจับให้ขึ้นแค่นี้
        setStatusState(i18nSafe("detecting", "กำลังจับท่า..."))
    }

    private fun stopRun(showSummary: Boolean) {
        isRunning = false
        imgToggle.setImageResource(android.R.drawable.ic_media_play)

        if (!showSummary) {
            setStatusState(i18nSafe("stopped", "หยุดแล้ว"))
            return
        }

        val lm = labelMap
        val a = agg
        if (lm == null || a == null) {
            setStatusState(i18nSafe("stopped", "หยุดแล้ว"))
            return
        }

        // ไม่สรุปเป็น no_action
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
            return
        }

        val key = lm[best.idx]
        // ✅ โชว์ชื่อท่าเฉพาะตอน “ผลลัพธ์/สรุป”
        setStatusState("${i18nSafe("summary", "ผลลัพธ์")}: ${i18n.t(key)}")
    }

    private fun findLabelIndex(lm: LabelMap, target: String): Int? {
        for (i in 0 until lm.size) if (lm[i] == target) return i
        return null
    }

    // =========================
    // Core: ระหว่างจับ “ไม่โชว์ชื่อท่า”
    // =========================
    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        if (!isRunning) return

        // 1) ไม่พบคน -> แจ้งเตือน + Pause (ไม่ Reset Buffer)
        if (!hasPerson(result)) {
            // seq.reset() <--- เอาออก (Pause)
            setStatusState(i18nSafe("no_person", "ไม่พบคน (หยุดชั่วคราว)"))
            return
        }

        // 2) พบคนแต่แขนไม่ครบ -> แจ้งเตือน + Pause (ไม่ Reset Buffer)
        if (!hasBothArms(result)) {
            // seq.reset() <--- เอาออก (Pause)
            setStatusState(i18nSafe("need_both_arms", "กรุณาให้เห็นแขนทั้ง 2 ข้าง"))
            return
        }

        // 3) เจอคน + แขนครบ -> ต้องกลับไป “กำลังจับท่า...”
        setStatusState(i18nSafe("detecting", "กำลังจับท่า..."))

        // ---- pipeline ปกติ (สะสมผลอย่างเดียว ไม่โชว์ชื่อท่า) ----
        val frame = Holistic258Extractor.extract(result)
        seq.add(frame)
        if (!seq.isFull()) return

        val probs = classifier?.predict(seq.toFlatFloatArray()) ?: return
        val top = top1top2(probs)

        val lm = labelMap ?: return
        val key = lm[top.topIdx]

        val rule = thr.forLabel(key)
        val diff = top.topScore - top.secondScore
        val pass = (top.topScore >= rule.tau) && (diff >= rule.delta)

        if (pass) {
            // ✅ ไม่สะสม no_action (เพราะไม่อยากให้ผลลัพธ์ออกเป็น no_action)
            if (key != "no_action") {
                agg?.add(top.topIdx, top.topScore)
            }
            // ❌ ไม่ setStatus เป็นชื่อท่าในระหว่างจับ (ตาม Logic หน้ากล้องปกติ)
        }
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

    // ต้องเห็นไหล่+ศอก+ข้อมือ ทั้ง 2 ข้าง
    private fun hasBothArms(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.size < 17) return false

        fun ok(idx: Int, visTh: Float): Boolean {
            val lm = pose[idx]
            val vis = try { lm.visibility().orElse(0f) } catch (_: Throwable) { 0f }
            val inFrame = lm.x() in 0f..1f && lm.y() in 0f..1f
            return inFrame && vis >= visTh
        }

        val vS = 0.35f
        val vE = 0.35f
        val vW = 0.35f

        val leftOk = ok(11, vS) && ok(13, vE) && ok(15, vW)
        val rightOk = ok(12, vS) && ok(14, vE) && ok(16, vW)
        return leftOk && rightOk
    }

    // ===== CameraX 1280x720 =====
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

        val preview = Preview.Builder()
            .setTargetResolution(Size(1280, 720))
            .build()
            .apply { setSurfaceProvider(previewView.surfaceProvider) }

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val r = runner
        if (r != null) {
            analysis.setAnalyzer(
                cameraExecutor,
                HolisticFrameAnalyzer(
                    enabled = { true },
                    runner = r
                )
            )
        }

        provider.unbindAll()
        provider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    // ✅ เปลี่ยนข้อความเฉพาะตอน “สถานะเปลี่ยน” (กันกระพริบ/สั่น)
    private fun setStatusState(msg: String) {
        if (msg == lastUiState) return
        lastUiState = msg
        runOnUiThread { if (::txtResult.isInitialized) txtResult.text = msg }
    }

    private fun i18nSafe(key: String, fallback: String): String {
        if (!::i18n.isInitialized) return fallback
        val v = try { i18n.t(key) } catch (_: Throwable) { key }
        return if (v == key) fallback else v
    }

    private fun assetExists(name: String): Boolean {
        return try { assets.open(name).close(); true } catch (_: Throwable) { false }
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
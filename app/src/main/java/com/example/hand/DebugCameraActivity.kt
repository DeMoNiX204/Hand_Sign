package com.example.hand

import android.Manifest
import android.os.Bundle
import android.os.SystemClock
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
import kotlin.math.min

class DebugCameraActivity : AppCompatActivity() {

    private val kLang = "th"
    private lateinit var i18n: I18n

    // ===== Switch Camera =====
    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val cameraSelector: CameraSelector
        get() = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    private var cameraProvider: ProcessCameraProvider? = null

    // UI
    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var txtDebug: TextView
    private lateinit var btnToggle: FrameLayout
    private lateinit var imgToggle: ImageView
    private lateinit var btnBack: ImageButton
    private lateinit var btnSwitchCamera: ImageButton

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var isRunning = false

    // ML
    private var runner: HolisticLandmarkerRunner? = null
    private var classifier: LitertSequenceClassifier? = null
    private var labelMap: LabelMap? = null
    private var thr: ThresholdConfig = ThresholdConfig()
    private var agg: PredictionAggregator? = null

    private val seq = SequenceBuffer(seqLen = ModelConfig.SEQ_T, featDim = ModelConfig.FEAT_F)
    private var seqCount = 0

    // Stats
    private var fpsStartTs = 0L
    private var fpsFrames = 0
    private var fps = 0f
    private var lastUiTs = 0L

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else setStatus(i18nSafe("camera_permission_denied", "ไม่ได้รับสิทธิ์กล้อง"))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_debug_camera)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // 1. Init UI First
        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        txtDebug = findViewById(R.id.txtDebug)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)

        btnBack.setOnClickListener { stopRun(false); finish() }
        btnSwitchCamera.setOnClickListener { switchCamera() }

        // 2. Check Assets
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatus("❌ Assets Missing:\n${missing.joinToString("\n")}")
            return
        }

        // 3. Load i18n safely
        try {
            i18n = I18n.fromAssets(this, kLang)
        } catch (e: Exception) {
            setStatus("❌ i18n fail: ${e.message}")
            return
        }

        // 4. Init Models
        try {
            labelMap = LabelMap.fromAssets(this, ModelConfig.MODEL_LABELS_ASSET)
            thr = ThresholdConfig.fromAssets(this, ModelConfig.MODEL_THRESH_ASSET)
            agg = PredictionAggregator(numClasses = labelMap!!.size)

            classifier = LitertSequenceClassifier(
                this,
                ModelConfig.MODEL_TFLITE_ASSET,
                labelMap!!.size
            )
        } catch (t: Throwable) {
            setStatus("❌ Init Fail: ${t.message}")
            return
        }

        try {
            runner = HolisticLandmarkerRunner(this, ModelConfig.HOLISTIC_TASK, false,
                onResult = { r -> onHolisticResult(r) },
                onError = { setStatus("❌ ML Error: $it") }
            )
        } catch (t: Throwable) {
            setStatus("❌ Runner Fail")
            return
        }

        btnToggle.setOnClickListener { if (!isRunning) startRun() else stopRun(true) }

        setStatus("✅ พร้อม (กด Play)")
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        seqCount = 0
        agg?.reset()
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)
        fpsStartTs = SystemClock.elapsedRealtime()
        fpsFrames = 0
        setStatus("กำลังจับท่า...")
    }

    private fun stopRun(showSummary: Boolean) {
        isRunning = false
        imgToggle.setImageResource(android.R.drawable.ic_media_play)
        if (!showSummary) { setStatus("หยุด"); return }

        val lm = labelMap ?: return
        val best = agg?.bestResultExclude(null, 0, Int.MAX_VALUE, 0f, Float.MAX_VALUE, 0f, Float.MAX_VALUE)
        if (best != null) {
            setStatus("ผลสรุป: ${i18n.t(lm[best.idx])}")
        } else {
            setStatus("หยุด (ยังไม่มีผลสรุป)")
        }
    }

    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        fpsFrames++
        val now = SystemClock.elapsedRealtime()
        if (now - fpsStartTs >= 1000) {
            fps = fpsFrames * 1000f / (now - fpsStartTs).toFloat()
            fpsFrames = 0
            fpsStartTs = now
        }

        // --- เตรียมตัวแปรสำหรับ Debug ---
        var top: Top2? = null
        var pass = false
        var usedLabel: String? = null
        var usedTau: Float? = null
        var usedDelta: Float? = null

        val personOk = hasPerson(result)
        val armsOk = hasBothArms(result)

        if (!isRunning) {
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta)
            return
        }

        if (!personOk) {
            setStatus("ไม่พบคน")
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta)
            return
        }

        if (!armsOk) {
            setStatus("เห็นแขนไม่ครบ")
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta)
            return
        }

        val frame = Holistic258Extractor.extract(result)
        seq.add(frame)
        seqCount = min(seqCount + 1, ModelConfig.SEQ_T)

        if (seq.isFull()) {
            val probs = classifier?.predict(seq.toFlatFloatArray())
            if (probs != null && labelMap != null) {
                top = top1top2(probs)
                val key = labelMap!![top.topIdx]

                val rule = thr.forLabel(key)
                usedLabel = key
                usedTau = rule.tau
                usedDelta = rule.delta

                val diff = top.topScore - top.secondScore
                pass = (top.topScore >= rule.tau) && (diff >= rule.delta)

                if (pass) {
                    if (key != "no_action") agg?.add(top.topIdx, top.topScore)
                }

                if (key == "no_action") {
                    setStatus("DEBUG: no_action (${f2(top.topScore)})")
                } else {
                    setStatus("${i18n.t(key)} (${f2(top.topScore)})")
                }
            }
        } else {
            setStatus("เก็บข้อมูล... $seqCount/30")
        }

        maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta)
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

    private fun hasBothArms(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.size < 17) return false
        fun ok(idx: Int, visTh: Float): Boolean {
            val lm = pose[idx]
            val vis = try { lm.visibility().orElse(0f) } catch (_: Throwable) { 0f }
            val inFrame = lm.x() in 0f..1f && lm.y() in 0f..1f
            return inFrame && vis >= visTh
        }
        val vS = 0.35f; val vE = 0.35f; val vW = 0.35f
        val leftOk = ok(11, vS) && ok(13, vE) && ok(15, vW)
        val rightOk = ok(12, vS) && ok(14, vE) && ok(16, vW)
        return leftOk && rightOk
    }

    private fun maybeUpdateDebug(
        now: Long, personOk: Boolean, armsOk: Boolean, seqC: Int,
        top: Top2?, pass: Boolean,
        label: String?, tau: Float?, delta: Float?
    ) {
        if (now - lastUiTs < 100) return
        lastUiTs = now

        val sb = StringBuilder()
        sb.appendLine("FPS: ${f1(fps)}")
        sb.appendLine("Mode: ${if(lensFacing==CameraSelector.LENS_FACING_BACK)"Back" else "Front"}")
        sb.appendLine("Gate: Person=${if(personOk)"✅" else "❌"} Arms=${if(armsOk)"✅" else "❌"}")
        sb.appendLine("Buffer: $seqC / 30")

        if (top != null && labelMap != null) {
            val k1 = labelMap!![top.topIdx]
            val k2 = labelMap!![top.secondIdx]

            sb.appendLine("-----------------------")
            sb.appendLine("1) ${i18n.t(k1)} (${f2(top.topScore)})")
            sb.appendLine("2) ${i18n.t(k2)} (${f2(top.secondScore)})")

            if (label != null && tau != null && delta != null) {
                sb.appendLine("Rule: $label (t=${f2(tau)}, d=${f2(delta)})")
            }

            val diff = top.topScore - top.secondScore
            sb.appendLine("Diff: ${f2(diff)}  Pass: $pass")
        } else {
            sb.appendLine("-----------------------")
            sb.appendLine("Wait for prediction...")
        }

        setDebug(sb.toString())
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

    data class Top2(val topIdx: Int, val topScore: Float, val secondIdx: Int, val secondScore: Float)

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)
        try { Holistic258Extractor.setMirror(isFront) } catch(_:Throwable){}

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val r = runner
        if (r != null) {
            analysis.setAnalyzer(cameraExecutor, HolisticFrameAnalyzer({true}, r))
        }
        provider.unbindAll()
        provider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    private fun switchCamera() {
        val p = cameraProvider ?: return
        val newLens = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        try {
            if (p.hasCamera(CameraSelector.Builder().requireLensFacing(newLens).build())) {
                lensFacing = newLens
                seq.reset(); seqCount = 0
                bindUseCases()
            }
        } catch(_:Throwable){}
    }

    private fun setStatus(msg: String) = runOnUiThread { if(::txtResult.isInitialized) txtResult.text = msg }
    private fun setDebug(msg: String) = runOnUiThread { if(::txtDebug.isInitialized) txtDebug.text = msg }
    private fun i18nSafe(k: String, f: String) = if(::i18n.isInitialized) i18n.t(k).let{ if(it==k) f else it } else f
    private fun assetExists(n: String) = try { assets.open(n).close(); true } catch (_: Throwable) { false }
    private fun f1(x: Float) = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Float) = String.format(Locale.US, "%.2f", x)

    // แก้ไขในไฟล์ DebugCameraActivity.kt
    override fun onDestroy() {
        // 1. หยุดรับภาพจากกล้องก่อนเป็นอันดับแรก (สำคัญ)
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}

        // 2. ปิด Thread
        cameraExecutor.shutdown()

        // 3. ปิด Model (ใส่ try-catch กันพลาด)
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}

        super.onDestroy()
    }
}
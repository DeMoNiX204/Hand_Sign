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

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // ===== UI =====
    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var txtDebug: TextView
    private lateinit var btnToggle: FrameLayout
    private lateinit var imgToggle: ImageView
    private lateinit var btnBack: ImageButton
    private lateinit var btnSwitchCamera: ImageButton

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
    private var seqCount = 0

    // ===== EMA Smoothing =====
    private var prevProbs: FloatArray? = null
    private val alphaEMA = 0.2f // ค่า Alpha 0.2 (ตรงกับ Python)

    // ===== Stats =====
    private var fpsStartTs = 0L
    private var fpsFrames = 0
    private var fps = 0f
    private var lastUiTs = 0L

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else setStatusState("ไม่ได้รับอนุญาตให้ใช้กล้อง")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_debug_camera)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        // 1. Init UI
        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        txtDebug = findViewById(R.id.txtDebug)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)

        btnBack.setOnClickListener {
            stopRun(showSummary = false)
            finish()
        }
        btnSwitchCamera.setOnClickListener { switchCamera() }

        // 2. Check Assets
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

        setStatusState("✅ พร้อม (กดปุ่มเล่น)")
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        try { Holistic258Extractor.reset() } catch (_: Throwable) {}
        seqCount = 0
        agg?.reset()
        prevProbs = null // รีเซ็ต EMA เมื่อเริ่มใหม่
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)

        fpsStartTs = SystemClock.elapsedRealtime()
        fpsFrames = 0

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

        // 🔥 Logic เดียวกับ CameraActivity
        val best = a.bestResultExclude(
            excludeIdx = noActionIdx,
            minCount = ModelConfig.MIN_COUNT_TO_ACCEPT,
            minSum = ModelConfig.MIN_SUM_TO_ACCEPT,
            minAvg = ModelConfig.MIN_AVG_SCORE_TO_ACCEPT
        )

        if (best == null) {
            setStatusState("ผลสรุป: ยังไม่มั่นใจ ลองใหม่")
        } else {
            val key = lm[best.idx]
            val percent = (best.avg * 100).toInt()
            setStatusState("ผลสรุป: ${i18n.t(key)} ($percent%)")
        }
    }

    // ===== Core Logic =====
    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        fpsFrames++
        val now = SystemClock.elapsedRealtime()
        if (now - fpsStartTs >= 1000) {
            fps = fpsFrames * 1000f / (now - fpsStartTs).toFloat()
            fpsFrames = 0
            fpsStartTs = now
        }

        if (!isRunning) return
        if (labelMap == null || classifier == null) return

        val lhCount = result.leftHandLandmarks().size
        val rhCount = result.rightHandLandmarks().size

        // 🔥 กฎเช็คคนเดียวกับ CameraActivity
        if (!hasPerson(result)) {
            setStatusState("ไม่พบคน")
            maybeUpdateDebug(now, false, hasBothArms(result), seqCount, null, false, null, null, null, lhCount, rhCount)
            return
        }
        if (!hasBothArms(result)) {
            setStatusState("เห็นแขนไม่ครบ")
            maybeUpdateDebug(now, true, false, seqCount, null, false, null, null, null, lhCount, rhCount)
            return
        }

        // ❌ เอา setStatusState("กำลังจับท่า...") ที่เคยอยู่ตรงนี้ออก เพื่อไม่ให้บังผล Real-time

        val frame = Holistic258Extractor.extract(result)
        val dataStr = frame.take(5).joinToString(", ") { f2(it) }

        seq.add(frame)
        seqCount = min(seqCount + 1, ModelConfig.SEQ_T)

        // ถ้ายูเซอร์เพิ่งกดเริ่ม และ Buffer ยังไม่เต็ม 30 เฟรม
        if (!seq.isFull()) {
            setStatusState("กำลังเก็บข้อมูล... ($seqCount/30)")
            maybeUpdateDebug(now, true, true, seqCount, null, false, null, null, dataStr, lhCount, rhCount)
            return
        }

        // 1. ได้ค่าดิบ (Raw Probabilities)
        val rawProbs = classifier?.predict(seq.getStartIndex()) ?: return

        // 3. คำนวณ EMA Smoothing
        val probs = if (prevProbs == null) {
            rawProbs
        } else {
            FloatArray(rawProbs.size) { i ->
                alphaEMA * rawProbs[i] + (1f - alphaEMA) * prevProbs!![i]
            }
        }
        prevProbs = probs

        // 4. หา Top 1 และ Top 2
        val top = top1top2(probs)
        val key = labelMap!![top.topIdx]

        val rule = thr.forLabel(key)

        // คำนวณความห่างและเช็คผ่านเกณฑ์
        val pass = (top.topScore >= rule.tau)

        if (pass && key != "no_action") {
            agg?.add(top.topIdx, top.topScore)
        }

        // ✅ อัปเดต UI แบบ Real-time ทันที!
        val livePercent = (top.topScore * 100).toInt()
        if (key == "no_action") {
            setStatusState("DEBUG: no_action ($livePercent%)")
        } else {
            val passStr = if (pass) "✅" else "⏳" // ถ้าผ่านโชว์ติ๊กถูก ถ้าไม่ผ่านโชว์ทรายกำกับ
            setStatusState("${i18n.t(key)} ($livePercent%) $passStr")
        }

        maybeUpdateDebug(now, true, true, seqCount, top, pass, key, rule.tau, dataStr, lhCount, rhCount)
    }

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

    private fun switchCamera() {
        val p = cameraProvider ?: return
        val newLens = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        try {
            if (p.hasCamera(CameraSelector.Builder().requireLensFacing(newLens).build())) {
                lensFacing = newLens
                seq.reset(); seqCount = 0
                prevProbs = null
                bindUseCases()
            }
        } catch(_:Throwable){}
    }

    private fun setStatusState(msg: String) {
        if (msg == lastUiState) return
        lastUiState = msg
        runOnUiThread { if (::txtResult.isInitialized) txtResult.text = msg }
    }

    private fun maybeUpdateDebug(
        now: Long, personOk: Boolean, armsOk: Boolean, seqC: Int,
        top: Top2?, pass: Boolean,
        label: String?, tau: Float?,
        dataStr: String? = null,
        lhCount: Int = 0, rhCount: Int = 0
    ) {
        if (now - lastUiTs < 100) return
        lastUiTs = now

        val sb = StringBuilder()
        sb.append("FPS: ${f1(fps)} | Buf: $seqC/30\n")

        val lIcon = if (lhCount > 0) "✅" else "❌"
        val rIcon = if (rhCount > 0) "✅" else "❌"
        sb.append("Hands: L$lIcon R$rIcon\n")

        if (dataStr != null) {
            sb.append("Data: $dataStr\n")
        }
        sb.append("----------------\n")

        if (top != null && labelMap != null) {
            val k1 = labelMap!![top.topIdx]
            val sc1 = (top.topScore * 100).toInt()
            val k2 = labelMap!![top.secondIdx]
            val sc2 = (top.secondScore * 100).toInt()

            sb.append("1) ${i18n.t(k1)} ($sc1%)\n")
            if (sc2 > 1) sb.append("2) ${i18n.t(k2)} ($sc2%)\n")

            if (label != null && tau != null ) {
                val pStr = if (pass) "Pass" else "Wait"
                sb.append("Rule: $label ($pStr)")
            }
        } else {
            sb.append("Scanning...")
        }

        runOnUiThread { if (::txtDebug.isInitialized) txtDebug.text = sb.toString() }
    }

    private fun assetExists(n: String) = try { assets.open(n).close(); true } catch (_: Throwable) { false }
    private fun f1(x: Float) = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Float) = String.format(Locale.US, "%.2f", x)

    override fun onDestroy() {
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}
        cameraExecutor.shutdown()
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}
        super.onDestroy()
    }

    private data class Top2(val topIdx: Int, val topScore: Float, val secondIdx: Int, val secondScore: Float)
}
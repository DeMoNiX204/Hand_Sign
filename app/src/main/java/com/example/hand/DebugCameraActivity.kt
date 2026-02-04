package com.example.hand

import android.Manifest
import android.os.Bundle
import android.os.SystemClock
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

    // ===== switch camera state =====
    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val cameraSelector: CameraSelector
        get() = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    private var cameraProvider: ProcessCameraProvider? = null

    // ✅ mirrorX คุณจะปรับเองทีหลัง (ตอนนี้ fix false)
    private val mirrorX = true

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

    // debug stats
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

        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        txtDebug = findViewById(R.id.txtDebug)
        btnToggle = findViewById(R.id.btnToggle)
        imgToggle = findViewById(R.id.imgToggle)
        btnBack = findViewById(R.id.btnBack)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)

        i18n = I18n.fromAssets(this, kLang)

        btnBack.setOnClickListener {
            stopRun(showSummary = false)
            finish()
        }

        btnSwitchCamera.setOnClickListener {
            switchCamera()
        }

        // check assets
        val missing = buildList {
            if (!assetExists(ModelConfig.HOLISTIC_TASK)) add(ModelConfig.HOLISTIC_TASK)
            if (!assetExists(ModelConfig.MODEL_TFLITE_ASSET)) add(ModelConfig.MODEL_TFLITE_ASSET)
            if (!assetExists(ModelConfig.MODEL_LABELS_ASSET)) add(ModelConfig.MODEL_LABELS_ASSET)
            if (!assetExists(ModelConfig.MODEL_THRESH_ASSET)) add(ModelConfig.MODEL_THRESH_ASSET)
            if (!assetExists("i18n/$kLang.json")) add("i18n/$kLang.json")
        }
        if (missing.isNotEmpty()) {
            setStatus("❌ assets หาย: ${missing.joinToString(", ")}")
            setDebug("ASSETS_MISSING:\n" + missing.joinToString("\n"))
            return
        }

        // init model/labels/thresholds
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
            setDebug("INIT_FAIL:\n${t.stackTraceToString()}")
            return
        }

        // init holistic runner
        try {
            runner = HolisticLandmarkerRunner(
                context = this,
                modelAssetName = ModelConfig.HOLISTIC_TASK,
                mirrorX = mirrorX,
                onResult = { r -> onHolisticResult(r) },
                onError = { msg ->
                    setStatus("❌ Holistic error: $msg")
                    setDebug("HOLISTIC_ERROR: $msg")
                }
            )
        } catch (t: Throwable) {
            setStatus("❌ Holistic init fail: ${t.message}")
            setDebug("HOLISTIC_INIT_FAIL:\n${t.stackTraceToString()}")
            return
        }

        btnToggle.setOnClickListener {
            if (!isRunning) startRun() else stopRun(showSummary = true)
        }

        setStatus(i18nSafe("ready_press_play", "✅ พร้อม (กดปุ่มเล่น)"))
        setDebug(
            "model=${ModelConfig.MODEL_TFLITE_ASSET}\n" +
                    "labels=${ModelConfig.MODEL_LABELS_ASSET}\n" +
                    "thr=${ModelConfig.MODEL_THRESH_ASSET}\n" +
                    "holistic=${ModelConfig.HOLISTIC_TASK}\n" +
                    "shape=${ModelConfig.SEQ_T}x${ModelConfig.FEAT_F}\n" +
                    "accept: count[${ModelConfig.MIN_COUNT_TO_ACCEPT}..${ModelConfig.MAX_COUNT_TO_ACCEPT}] " +
                    "sum[${ModelConfig.MIN_SUM_TO_ACCEPT}..${ModelConfig.MAX_SUM_TO_ACCEPT}] " +
                    "avg[${ModelConfig.MIN_AVG_SCORE_TO_ACCEPT}..${ModelConfig.MAX_AVG_SCORE_TO_ACCEPT}]"
        )

        askCamera.launch(Manifest.permission.CAMERA)
    }

    // ===== RUN control =====
    private fun startRun() {
        isRunning = true
        seq.reset()
        seqCount = 0
        agg?.reset() // ✅ เริ่มใหม่ค่อยล้างผลสะสม
        imgToggle.setImageResource(android.R.drawable.ic_media_pause)

        fpsStartTs = SystemClock.elapsedRealtime()
        fpsFrames = 0
        fps = 0f
        lastUiTs = 0L

        setStatus(i18nSafe("detecting", "กำลังจับท่า..."))
    }

    // ✅ แบบ A: STOP ต้องห้ามสรุปเป็น no_action
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
            setStatus(i18nSafe("not_sure", "ยังไม่มั่นใจ ลองใหม่"))
            return
        }

        val key = lm[best.idx]
        setStatus("${i18nSafe("summary", "ผลสรุป")}: ${i18n.t(key)}")
        setDebug(
            "STOP BEST: idx=${best.idx} key=$key count=${best.count} sum=${f2(best.sum)} avg=${f2(best.avg)}\n\n" +
                    txtDebug.text
        )
    }

    private fun findLabelIndex(lm: LabelMap, target: String): Int? {
        for (i in 0 until lm.size) {
            if (lm[i] == target) return i
        }
        return null
    }

    // ===== result callback =====
    private fun onHolisticResult(result: HolisticLandmarkerResult) {
        // fps
        fpsFrames++
        val now = SystemClock.elapsedRealtime()
        val dt = now - fpsStartTs
        if (dt >= 1000L) {
            fps = fpsFrames * 1000f / dt.toFloat()
            fpsFrames = 0
            fpsStartTs = now
        }

        val gate = calcGate(result)
        val armsOk = hasBothArms(result) // ✅ เพิ่ม gate แขน 2 ข้าง

        if (!isRunning) {
            maybeUpdateDebug(now, gate, armsOk, null, null, null, false, null)
            return
        }

        // ✅ ไม่เจอคน: reset แค่ seq/seqCount (ไม่ล้าง agg)
        if (!gate.hasPerson) {
            seq.reset()
            seqCount = 0
            setStatus(i18nSafe("no_person", "ไม่พบคน"))
            maybeUpdateDebug(now, gate, armsOk, null, null, null, false, null)
            return
        }

        // ✅ ไม่เจอแขน 2 ข้าง: reset แค่ seq/seqCount (ไม่ล้าง agg)
        if (!armsOk) {
            seq.reset()
            seqCount = 0
            setStatus(i18nSafe("need_both_arms", "กรุณาให้เห็นแขนทั้ง 2 ข้าง"))
            maybeUpdateDebug(now, gate, armsOk, null, null, null, false, null)
            return
        }

        val frame = Holistic258Extractor.extract(result)
        val nonZero = frame.count { it != 0f }
        val first12 = frame.take(12).joinToString(", ") { f2(it) }

        seq.add(frame)
        seqCount = min(seqCount + 1, ModelConfig.SEQ_T)

        var top: Top2? = null
        var pass = false
        var top5: List<Pair<Int, Float>>? = null

        if (seq.isFull()) {
            val probs = classifier?.predict(seq.toFlatFloatArray())
            if (probs != null) {
                top = top1top2(probs)
                pass = top.topScore >= thr.tau && (top.topScore - top.secondScore) >= thr.delta
                top5 = topK(probs, 5)
                if (pass) agg?.add(top.topIdx, top.topScore)

                if (pass) {
                    val key = labelMap?.get(top.topIdx) ?: "class_${top.topIdx}"
                    setStatus("${i18n.t(key)} (${f2(top.topScore)})")
                } else {
                    setStatus("...")
                }
            }
        }

        maybeUpdateDebug(now, gate, armsOk, nonZero, first12, top, pass, top5)
    }

    // ===== Gate =====
    private data class GateInfo(
        val hasPerson: Boolean,
        val poseCount: Int,
        val leftHandCount: Int,
        val rightHandCount: Int,
        val meanVis: Float,
        val goodVisCount: Int
    )

    private fun calcGate(r: HolisticLandmarkerResult): GateInfo {
        val pose = r.poseLandmarks()
        val poseCount = pose.size
        val left = try { r.leftHandLandmarks().size } catch (_: Throwable) { 0 }
        val right = try { r.rightHandLandmarks().size } catch (_: Throwable) { 0 }

        if (poseCount == 0) return GateInfo(false, 0, left, right, 0f, 0)

        var sumVis = 0f
        var good = 0
        for (lm in pose) {
            val vis = try { lm.visibility().orElse(0f) } catch (_: Throwable) { 0f }
            sumVis += vis
            if (vis >= 0.5f) good++
        }
        val meanVis = sumVis / poseCount.toFloat()
        val has = meanVis >= 0.25f && good >= 8
        return GateInfo(has, poseCount, left, right, meanVis, good)
    }

    // ✅ เพิ่มใหม่: ต้องเห็น “ไหล่+ศอก+ข้อมือ” ทั้งซ้ายและขวา
    // Pose index: L shoulder=11, L elbow=13, L wrist=15 | R shoulder=12, R elbow=14, R wrist=16
    private fun hasBothArms(r: HolisticLandmarkerResult): Boolean {
        val pose = r.poseLandmarks()
        if (pose.size < 17) return false

        fun ok(idx: Int, visTh: Float): Boolean {
            val lm = pose[idx]
            val vis = try { lm.visibility().orElse(0f) } catch (_: Throwable) { 0f }
            val inFrame = lm.x() in 0f..1f && lm.y() in 0f..1f
            return inFrame && vis >= visTh
        }

        // ถ้าไม่ผ่านง่ายไป ลดเป็น 0.25f ได้
        val vS = 0.35f
        val vE = 0.35f
        val vW = 0.35f

        val leftOk = ok(11, vS) && ok(13, vE) && ok(15, vW)
        val rightOk = ok(12, vS) && ok(14, vE) && ok(16, vW)
        return leftOk && rightOk
    }

    // ===== Debug UI =====
    private fun maybeUpdateDebug(
        now: Long,
        gate: GateInfo,
        armsOk: Boolean,
        nonZero: Int?,
        first12: String?,
        top: Top2?,
        pass: Boolean,
        top5: List<Pair<Int, Float>>?
    ) {
        if (now - lastUiTs < 100L) return
        lastUiTs = now

        val sb = StringBuilder()
        sb.appendLine("FPS: ${f1(fps)}")
        sb.appendLine("RUN: $isRunning   SEQ: $seqCount/${ModelConfig.SEQ_T}")
        sb.appendLine("CAM: ${if (lensFacing == CameraSelector.LENS_FACING_BACK) "BACK" else "FRONT"}")
        sb.appendLine("Gate: hasPerson=${gate.hasPerson}  armsOk=$armsOk")
        sb.appendLine("pose=${gate.poseCount}  Lhand=${gate.leftHandCount}  Rhand=${gate.rightHandCount}")
        sb.appendLine("meanVis=${f2(gate.meanVis)}  goodVis=${gate.goodVisCount}")
        sb.appendLine("thr: tau=${f2(thr.tau)}  delta=${f2(thr.delta)}")

        if (nonZero != null) sb.appendLine("feat nonZero=$nonZero / ${ModelConfig.FEAT_F}")
        if (first12 != null) sb.appendLine("feat[0..11]=[$first12]")

        val lm = labelMap
        if (top != null && lm != null) {
            val k1 = lm[top.topIdx]
            val k2 = lm[top.secondIdx]
            sb.appendLine("top1: ${i18n.t(k1)} ($k1) = ${f2(top.topScore)}")
            sb.appendLine("top2: ${i18n.t(k2)} ($k2) = ${f2(top.secondScore)}")
            sb.appendLine("pass=$pass diff=${f2(top.topScore - top.secondScore)}")

            if (top5 != null) {
                sb.appendLine("top5:")
                for ((rank, pair) in top5.withIndex()) {
                    val idx = pair.first
                    val sc = pair.second
                    val key = lm[idx]
                    sb.appendLine("  ${rank + 1}) ${i18n.t(key)} ($key) = ${f2(sc)}")
                }
            }
        } else {
            sb.appendLine("top1: -")
        }

        setDebug(sb.toString())
    }

    private fun topK(probs: FloatArray, k: Int): List<Pair<Int, Float>> {
        val idx = probs.indices.toMutableList()
        idx.sortByDescending { probs[it] }
        return idx.take(k).map { it to probs[it] }
    }

    // ===== CameraX =====
    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return

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
                    enabled = { true },
                    runner = r
                )
            )
        }

        provider.unbindAll()
        provider.bindToLifecycle(this, cameraSelector, preview, analysis)

        if (!isRunning) {
            val camName = if (lensFacing == CameraSelector.LENS_FACING_BACK) "กล้องหลัง" else "กล้องหน้า"
            setStatus("✅ $camName (กดปุ่มเล่น)")
        }
    }

    private fun switchCamera() {
        val provider = cameraProvider
        if (provider == null) {
            setStatus("⏳ รอกล้องพร้อมก่อน...")
            return
        }

        val newFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT
        else
            CameraSelector.LENS_FACING_BACK

        val newSelector = CameraSelector.Builder().requireLensFacing(newFacing).build()

        val ok = try { provider.hasCamera(newSelector) } catch (_: Throwable) { false }
        if (!ok) {
            setStatus("❌ ไม่มี ${if (newFacing == CameraSelector.LENS_FACING_BACK) "กล้องหลัง" else "กล้องหน้า"}")
            return
        }

        lensFacing = newFacing

        // ✅ กันเฟรมปน: reset แค่ seq/seqCount (ไม่ล้าง agg)
        seq.reset()
        seqCount = 0

        bindUseCases()
    }

    // ===== helpers =====
    private fun setStatus(msg: String) {
        runOnUiThread { if (::txtResult.isInitialized) txtResult.text = msg }
    }

    private fun setDebug(msg: String) {
        runOnUiThread { if (::txtDebug.isInitialized) txtDebug.text = msg }
    }

    private fun i18nSafe(key: String, fallback: String): String {
        if (!::i18n.isInitialized) return fallback
        val v = i18n.t(key)
        return if (v == key) fallback else v
    }

    private fun assetExists(name: String): Boolean {
        return try { assets.open(name).close(); true } catch (_: Throwable) { false }
    }

    private fun f1(x: Float) = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Float) = String.format(Locale.US, "%.2f", x)

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

    // ===== Top2 helper =====
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

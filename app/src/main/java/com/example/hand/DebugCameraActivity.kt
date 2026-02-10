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

    // 🔥 1. กำหนด mirrorX = true เพื่อให้พิกัดตรงกับ Python (ทั้งหน้าและหลัง)
    // ลองเปลี่ยนเป็น true/false ดูถ้าค่ามันกลับด้าน
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

    // 🔥 2. เพิ่มตัวแปรสำหรับ EMA Smoothing
    private var prevProbs: FloatArray? = null
    private val alphaEMA = 0.2f

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
            // 🔥 ส่ง mirrorX เข้าไปที่ Runner
            runner = HolisticLandmarkerRunner(this, ModelConfig.HOLISTIC_TASK, mirrorX,
                onResult = { r -> onHolisticResult(r) },
                onError = { setStatus("❌ ML Error: $it") }
            )
        } catch (t: Throwable) {
            setStatus("❌ Runner Fail")
            return
        }

        // ปุ่ม Start
        btnToggle.setOnClickListener { if (!isRunning) startRun() else stopRun(true) }

        setStatus("✅ พร้อม (กด Play)")
        askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startRun() {
        isRunning = true
        seq.reset()
        seqCount = 0
        agg?.reset()
        prevProbs = null // รีเซ็ต EMA
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

        var top: Top2? = null
        var pass = false
        var usedLabel: String? = null
        var usedTau: Float? = null
        var usedDelta: Float? = null

        // 1. เช็คว่าเจอคนไหม
        val personOk = hasPerson(result)
        val armsOk = hasBothArms(result)

        // 🔥 2. เช็คว่าเจอมือไหม (นี่คือจุดสำคัญ!)
        val lhCount = result.leftHandLandmarks().size
        val rhCount = result.rightHandLandmarks().size

        // ถ้าไม่ได้กด Start หรือคนไม่ครบ ก็ให้อัปเดตหน้าจอ Debug ด้วย
        if (!isRunning || !personOk || !armsOk) {
            val statusMsg = if (!personOk) "ไม่พบคน" else if (!armsOk) "เห็นแขนไม่ครบ" else null
            if (statusMsg != null) setStatus(statusMsg)

            // ส่งจำนวนมือไปแสดงผลด้วย
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta, null, lhCount, rhCount)
            return
        }

        // 3. สกัด Feature
        val frame = Holistic258Extractor.extract(result)

        // สร้าง String 5 ตัวแรกเพื่อดู Debug
        val dataStr = frame.take(5).joinToString(", ") { f2(it) }
        android.util.Log.d("MP_DEBUG", "Frame: $dataStr | L:$lhCount R:$rhCount")

        // 4. เข้า Buffer
        seq.add(frame)
        seqCount = min(seqCount + 1, ModelConfig.SEQ_T)

        if (seq.isFull()) {
            val rawProbs = classifier?.predict(seq.toFlatFloatArray())
            if (rawProbs != null && labelMap != null) {

                // EMA Smoothing
                val probs = if (prevProbs == null) {
                    rawProbs
                } else {
                    FloatArray(rawProbs.size) { i ->
                        alphaEMA * rawProbs[i] + (1f - alphaEMA) * prevProbs!![i]
                    }
                }
                prevProbs = probs

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

            // ส่งข้อมูลไปแสดงผล (รวมถึงจำนวนมือ)
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta, dataStr, lhCount, rhCount)

        } else {
            setStatus("เก็บข้อมูล... $seqCount/30")
            maybeUpdateDebug(now, personOk, armsOk, seqCount, top, pass, usedLabel, usedTau, usedDelta, dataStr, lhCount, rhCount)
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

    // ฟังก์ชันอัปเดต UI พร้อมพารามิเตอร์ dataStr
    private fun maybeUpdateDebug(
        now: Long, personOk: Boolean, armsOk: Boolean, seqC: Int,
        top: Top2?, pass: Boolean,
        label: String?, tau: Float?, delta: Float?,
        dataStr: String? = null,
        lhCount: Int = 0, rhCount: Int = 0 // รับค่าจำนวนมือ
    ) {
        if (now - lastUiTs < 100) return
        lastUiTs = now

        val sb = StringBuilder()
        sb.append("FPS: ${f1(fps)} | Buf: $seqC/30\n")

        // 🔥 แสดงสถานะมือซ้าย/ขวา
        // ถ้าเป็น 0 แปลว่าไม่เจอมือ (กากบาท), ถ้ามากกว่า 0 แปลว่าเจอ (ติ๊กถูก)
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

            if (label != null && tau != null && delta != null) {
                val pStr = if (pass) "Pass" else "Wait"
                sb.append("Rule: $label ($pStr)")
            }
        } else {
            sb.append("Scanning...")
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

        // 🔥 เปลี่ยนจาก isFront เป็น mirrorX เพื่อบังคับ Mirror ตาม Python
        try { Holistic258Extractor.setMirror(mirrorX) } catch(_:Throwable){}

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(720, 1280))
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
                prevProbs = null // รีเซ็ต EMA เมื่อสลับกล้อง
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

    override fun onDestroy() {
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}
        cameraExecutor.shutdown()
        try { runner?.close() } catch (_: Throwable) {}
        try { classifier?.close() } catch (_: Throwable) {}
        super.onDestroy()
    }
}
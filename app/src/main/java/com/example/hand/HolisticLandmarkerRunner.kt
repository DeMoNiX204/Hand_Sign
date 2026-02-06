package com.example.hand

import android.content.Context
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import java.nio.ByteBuffer

class HolisticLandmarkerRunner(
    context: Context,
    modelAssetName: String,
    private val mirrorX: Boolean,
    private val onResult: (HolisticLandmarkerResult) -> Unit,
    private val onError: (String) -> Unit
) : AutoCloseable {

    private var landmarker: HolisticLandmarker? = null
    private var isClosed = false

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath(modelAssetName)
            .build()

        val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setOutputFaceBlendshapes(false)
            .setOutputPoseSegmentationMasks(false)
            .setResultListener { result, input ->
                // ถ้าปิดไปแล้ว ไม่ต้องส่งผลลัพธ์กลับ
                if (isClosed) return@setResultListener

                try {
                    onResult(result)
                } catch (t: Throwable) {
                    onError(t.message ?: "onResult failed")
                }
                // ไม่ต้อง input.close() ที่นี่เพราะ MediaPipe จัดการให้ใน Live Stream Mode (แต่ถ้าใส่ก็ได้)
            }
            .setErrorListener { e ->
                if (!isClosed) onError(e.message ?: "HolisticLandmarker error")
            }
            .build()

        landmarker = HolisticLandmarker.createFromOptions(context, options)
        Holistic258Extractor.setMirror(mirrorX)
    }

    fun detectAsync(
        rgbaBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        timestampMs: Long
    ) {
        if (isClosed || landmarker == null) return

        try {
            val mpImage = ByteBufferImageBuilder(
                rgbaBuffer,
                width,
                height,
                MPImage.IMAGE_FORMAT_RGBA
            ).build()

            val opt = ImageProcessingOptions.builder()
                .setRotationDegrees(rotationDegrees)
                .build()

            // 🔥 ใส่ try-catch ป้องกัน Crash ถ้าเผลอเรียกตอนปิดแอป
            landmarker?.detectAsync(mpImage, opt, timestampMs)

        } catch (e: Exception) {
            // กิน Error ทิ้งไปเลยถ้ามันพังตอนปิด
            if (!isClosed) {
                onError("Detect error: ${e.message}")
            }
        }
    }

    override fun close() {
        isClosed = true
        try {
            landmarker?.close()
        } catch (_: Throwable) {}
        landmarker = null
    }
}
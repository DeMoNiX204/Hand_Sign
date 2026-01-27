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

    private val landmarker: HolisticLandmarker

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath(modelAssetName) // "holistic_landmarker.task"
            .build()

        val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setOutputFaceBlendshapes(false)
            .setOutputPoseSegmentationMasks(false)
            .setResultListener { result, input ->
                try {
                    // ส่ง result ให้ต่อไปสกัด 258
                    onResult(result)
                } catch (t: Throwable) {
                    onError(t.message ?: "onResult failed")
                } finally {
                    try { input.close() } catch (_: Throwable) {}
                }
            }
            .setErrorListener { e ->
                onError(e.message ?: "HolisticLandmarker error")
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
        val mpImage = ByteBufferImageBuilder(
            rgbaBuffer,
            width,
            height,
            MPImage.IMAGE_FORMAT_RGBA
        ).build()

        val opt = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()

        landmarker.detectAsync(mpImage, opt, timestampMs)
    }

    override fun close() {
        landmarker.close()
    }
}

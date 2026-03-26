package com.example.hand

import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

class HolisticFrameAnalyzer(
    private val enabled: () -> Boolean,
    private val runner: HolisticLandmarkerRunner
) : ImageAnalysis.Analyzer {

    private var lastTs = 0L

    override fun analyze(image: ImageProxy) {
        if (!enabled()) {
            image.close()
            return
        }

        try {
            val plane = image.planes[0]
            val width  = image.width
            val height = image.height

            val rgba = packRgba(
                src         = plane.buffer,
                width       = width,
                height      = height,
                rowStride   = plane.rowStride,
                pixelStride = plane.pixelStride
            )

            val now = SystemClock.uptimeMillis()
            val ts  = max(now, lastTs + 1)
            lastTs  = ts


            runner.detectAsync(
                rgbaBuffer     = rgba,
                width          = width,
                height         = height,
                rotationDegrees = 0,
                timestampMs    = ts
            )
        } finally {
            image.close()
        }
    }

    private fun packRgba(
        src: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int
    ): ByteBuffer {
        val out = ByteBuffer.allocateDirect(width * height * 4)
            .order(ByteOrder.nativeOrder())

        val rowBytes = width * pixelStride
        val tmp      = ByteArray(rowBytes)
        val buf      = src.duplicate()

        for (y in 0 until height) {
            buf.position(y * rowStride)
            buf.get(tmp, 0, rowBytes)
            out.put(tmp)
        }

        out.rewind()
        return out
    }
}
package com.example.hand

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

object Holistic258Extractor {

    private var mirrorX: Boolean = false
    fun setMirror(mirror: Boolean) { mirrorX = mirror }

    fun extract(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(258)
        var k = 0

        // Pose: 33 * (x,y,z,visibility) = 132
        val pose = r.poseLandmarks()
        for (i in 0 until 33) {
            val lm = pose.getOrNull(i)
            val x = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
            val y = lm?.y() ?: 0f
            val z = lm?.z() ?: 0f
            val vis = lm?.visibility()?.orElse(0f) ?: 0f

            out[k++] = x
            out[k++] = y
            out[k++] = z
            out[k++] = vis
        }

        // Left hand: 21 * (x,y,z) = 63
        val lh = r.leftHandLandmarks()
        for (i in 0 until 21) {
            val lm = lh.getOrNull(i)
            val x = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
            val y = lm?.y() ?: 0f
            val z = lm?.z() ?: 0f
            out[k++] = x
            out[k++] = y
            out[k++] = z
        }

        // Right hand: 21 * (x,y,z) = 63
        val rh = r.rightHandLandmarks()
        for (i in 0 until 21) {
            val lm = rh.getOrNull(i)
            val x = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
            val y = lm?.y() ?: 0f
            val z = lm?.z() ?: 0f
            out[k++] = x
            out[k++] = y
            out[k++] = z
        }

        return out
    }

    private fun List<NormalizedLandmark>.getOrNull(i: Int): NormalizedLandmark? {
        return if (i in indices) this[i] else null
    }
}

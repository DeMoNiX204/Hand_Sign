package com.example.hand

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlin.math.hypot

/**
 * 258-dim feature extractor.
 *
 * Layout (float32):
 * - Pose:       33 * (x, y, z, visibility) = 132
 * - Left hand:  21 * (x, y, z)            = 63
 * - Right hand: 21 * (x, y, z)            = 63
 *
 * Training-match normalization:
 * - Apply mirrorX on x first (for front camera, if you trained that way).
 * - Compute shoulder-mid using pose landmarks 11 (L shoulder) and 12 (R shoulder).
 * - Compute shoulder width = distance between (11) and (12) in (x,y).
 * - For every landmark coordinate: (x,y,z) -> ((x-cx)/s, (y-cy)/s, (z-cz)/s)
 * where (cx,cy,cz) is shoulder-mid and s = max(shoulderWidth, eps).
 *
 * Update:
 * - Fixed missing hand handling: if hand is not detected, fill 0.0 (instead of calculating with raw 0).
 */
object Holistic258Extractor {

    private const val EPS = 1e-6f

    private var mirrorX: Boolean = false
    fun setMirror(mirror: Boolean) { mirrorX = mirror }

    fun extract(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(258)
        var k = 0

        val pose = r.poseLandmarks()

        // ---- reference for normalization (shoulder mid + shoulder width) ----
        val lShoulder = pose.getOrNullSafe(11)
        val rShoulder = pose.getOrNullSafe(12)

        val lxs = lShoulder?.x()?.let { if (mirrorX) 1f - it else it }
        val lys = lShoulder?.y()
        val lzs = lShoulder?.z()

        val rxs = rShoulder?.x()?.let { if (mirrorX) 1f - it else it }
        val rys = rShoulder?.y()
        val rzs = rShoulder?.z()

        val ref = if (
            lxs != null && lys != null && lzs != null &&
            rxs != null && rys != null && rzs != null
        ) {
            val cx = (lxs + rxs) * 0.5f
            val cy = (lys + rys) * 0.5f
            val cz = (lzs + rzs) * 0.5f
            val shoulderW = hypot((lxs - rxs).toDouble(), (lys - rys).toDouble()).toFloat()
            val s = if (shoulderW > EPS) shoulderW else 1f
            Ref(cx, cy, cz, s)
        } else {
            // Fallback: no reliable shoulders -> don't normalize
            Ref(0f, 0f, 0f, 1f)
        }

        // 1) Pose: 33 * (x,y,z,visibility) = 132
        for (i in 0 until 33) {
            val lm = pose.getOrNullSafe(i)
            val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
            val rawY = lm?.y() ?: 0f
            val rawZ = lm?.z() ?: 0f
            val vis = lm?.visibility()?.orElse(0f) ?: 0f

            out[k++] = (rawX - ref.cx) / ref.s
            out[k++] = (rawY - ref.cy) / ref.s
            out[k++] = (rawZ - ref.cz) / ref.s
            out[k++] = vis
        }

        // 2) Left hand: 21 * (x,y,z) = 63
        val lh = r.leftHandLandmarks()
        if (lh.isEmpty()) {
            // ✅ FIX: ถ้าไม่มีมือ ให้ใส่ 0.0 ทั้งหมด (เพื่อให้เหมือน Python Zero Padding)
            repeat(21 * 3) { out[k++] = 0f }
        } else {
            for (i in 0 until 21) {
                val lm = lh.getOrNullSafe(i)
                val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
                val rawY = lm?.y() ?: 0f
                val rawZ = lm?.z() ?: 0f

                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s
            }
        }

        // 3) Right hand: 21 * (x,y,z) = 63
        val rh = r.rightHandLandmarks()
        if (rh.isEmpty()) {
            // ✅ FIX: ถ้าไม่มีมือ ให้ใส่ 0.0 ทั้งหมด
            repeat(21 * 3) { out[k++] = 0f }
        } else {
            for (i in 0 until 21) {
                val lm = rh.getOrNullSafe(i)
                val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
                val rawY = lm?.y() ?: 0f
                val rawZ = lm?.z() ?: 0f

                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s
            }
        }

        return out
    }

    private fun List<NormalizedLandmark>.getOrNullSafe(i: Int): NormalizedLandmark? {
        return if (i in indices) this[i] else null
    }

    private data class Ref(
        val cx: Float,
        val cy: Float,
        val cz: Float,
        val s: Float
    )
}
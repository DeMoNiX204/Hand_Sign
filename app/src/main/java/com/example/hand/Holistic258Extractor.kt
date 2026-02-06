package com.example.hand

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlin.math.hypot

object Holistic258Extractor {

    private const val EPS = 1e-6f

    // 🔥 บังคับเป็น true เสมอ เพื่อให้เหมือนภาพจาก Webcam (แก้ปัญหาซ้าย/ขวาสลับกัน)
    private var mirrorX: Boolean = true

    fun setMirror(mirror: Boolean) {
        // mirrorX = mirror // ❌ ปิดไว้ก่อน เพื่อบังคับเทสด้วยโหมด Mirror
        mirrorX = true
    }

    fun extract(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(258)
        var k = 0

        val pose = r.poseLandmarks()

        // ---- 1. หาจุดอ้างอิง (Reference) จากไหล่ ----
        val lShoulder = pose.getOrNullSafe(11)
        val rShoulder = pose.getOrNullSafe(12)

        // ถ้า mirrorX=true จะกลับค่า x เป็น (1 - x)
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
            val cz = (lzs + rzs) * 0.5f // ค่า Z เฉลี่ยของไหล่
            val shoulderW = hypot((lxs - rxs).toDouble(), (lys - rys).toDouble()).toFloat()
            val s = if (shoulderW > EPS) shoulderW else 1f
            Ref(cx, cy, cz, s)
        } else {
            Ref(0f, 0f, 0f, 1f)
        }

        // ---- 2. Pose (33 points) ----
        for (i in 0 until 33) {
            val lm = pose.getOrNullSafe(i)
            val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
            val rawY = lm?.y() ?: 0f
            val rawZ = lm?.z() ?: 0f
            val vis = lm?.visibility()?.orElse(0f) ?: 0f

            // สูตร Python: (val - center) / size
            out[k++] = (rawX - ref.cx) / ref.s
            out[k++] = (rawY - ref.cy) / ref.s

            // 🔥 แก้ไข: ต้องลบ ref.cz เพื่อให้ตรงกับ extractkeypoint.py บรรทัด 67
            out[k++] = (rawZ - ref.cz) / ref.s

            out[k++] = vis
        }

        // ---- 3. Left Hand (21 points) ----
        val lh = r.leftHandLandmarks()
        if (lh.isEmpty()) {
            repeat(21 * 3) { out[k++] = 0f }
        } else {
            for (i in 0 until 21) {
                val lm = lh.getOrNullSafe(i)
                val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
                val rawY = lm?.y() ?: 0f
                val rawZ = lm?.z() ?: 0f

                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s // 🔥 แก้ไข: ลบ ref.cz
            }
        }

        // ---- 4. Right Hand (21 points) ----
        val rh = r.rightHandLandmarks()
        if (rh.isEmpty()) {
            repeat(21 * 3) { out[k++] = 0f }
        } else {
            for (i in 0 until 21) {
                val lm = rh.getOrNullSafe(i)
                val rawX = lm?.x()?.let { if (mirrorX) 1f - it else it } ?: 0f
                val rawY = lm?.y() ?: 0f
                val rawZ = lm?.z() ?: 0f

                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s // 🔥 แก้ไข: ลบ ref.cz
            }
        }

        return out
    }

    private fun List<NormalizedLandmark>.getOrNullSafe(i: Int): NormalizedLandmark? {
        return if (i in indices) this[i] else null
    }

    private data class Ref(
        val cx: Float, val cy: Float, val cz: Float, val s: Float
    )
}
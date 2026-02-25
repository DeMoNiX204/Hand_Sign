package com.example.hand

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlin.math.hypot

object Holistic258Extractor {

    // ==========================================
    // 🎛️ CONFIG ZONE (ไม่แตะต้องตามที่คุณขอ)
    // ==========================================
    private const val SWAP_AXES = true
    private const val INVERT_X  = false
    private const val INVERT_Y  = false

    // 🔥 ปรับให้ตรงกับ Python (0.001)
    private const val MIN_BODY_SIZE = 0.001f

    // 💾 ระบบจำค่าล่าสุด (Forward Fill) แบบเดียวกับ Python ป้องกันมือวาร์ป
    private var prevPose = FloatArray(132)
    private var prevRightHand = FloatArray(63)
    private var prevLeftHand = FloatArray(63)

    // ✅ เรียกใช้ตอนสลับกล้องหรือเริ่มจับภาพใหม่ เพื่อล้างความจำมือ
    fun reset() {
        prevPose = FloatArray(132)
        prevRightHand = FloatArray(63)
        prevLeftHand = FloatArray(63)
    }

    fun setMirror(mirror: Boolean) { }

    fun extract(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(258)
        var k = 0
        val pose = r.poseLandmarks()

        fun getLandmarkXY(lm: NormalizedLandmark?): Pair<Float, Float> {
            if (lm == null) return Pair(0.5f, 0.5f)
            var x = lm.x()
            var y = lm.y()
            if (SWAP_AXES) { val temp = x; x = y; y = temp }
            if (INVERT_X) x = 1f - x
            if (INVERT_Y) y = 1f - y
            return Pair(x, y)
        }

        fun getLandmarkZ(lm: NormalizedLandmark?) = lm?.z() ?: 0f
        fun getVis(lm: NormalizedLandmark?) = lm?.visibility()?.orElse(0f) ?: 0f

        val swapMap = mapOf(
            1 to 4, 4 to 1, 2 to 5, 5 to 2, 3 to 6, 6 to 3, 7 to 8, 8 to 7, 9 to 10, 10 to 9,
            11 to 12, 12 to 11, 13 to 14, 14 to 13, 15 to 16, 16 to 15,
            17 to 18, 18 to 17, 19 to 20, 20 to 19, 21 to 22, 22 to 21,
            23 to 24, 24 to 23, 25 to 26, 26 to 25, 27 to 28, 28 to 27, 29 to 30, 30 to 29, 31 to 32, 32 to 31
        )

        // 1. คำนวณ Scale และ Ref (สูตรเดียวกับ Python 100%)
        val lShoulderRaw = getOrNullSafe(pose, swapMap[11] ?: 11)
        val rShoulderRaw = getOrNullSafe(pose, swapMap[12] ?: 12)

        val (lxs, lys) = getLandmarkXY(lShoulderRaw)
        val (rxs, rys) = getLandmarkXY(rShoulderRaw)
        val lzs = getLandmarkZ(lShoulderRaw)
        val rzs = getLandmarkZ(rShoulderRaw)

        val refX: Float; val refY: Float; val refZ: Float; val scale: Float

        if (lShoulderRaw != null && rShoulderRaw != null) {
            val dw = hypot((lxs - rxs).toDouble(), (lys - rys).toDouble()).toFloat()
            // 🔥 ปรับสูตรเช็ค Body Size ให้ตรง Python (เช็ค < 0.001)
            scale = if (dw < MIN_BODY_SIZE) 1f else dw

            refX = (lxs + rxs) * 0.5f
            refY = (lys + rys) * 0.5f
            refZ = (lzs + rzs) * 0.5f
        } else {
            refX = 0.5f; refY = 0.5f; refZ = 0f; scale = 1f
        }

        fun feat(x: Float, y: Float, z: Float): Triple<Float, Float, Float> {
            return Triple((x - refX) / scale, (y - refY) / scale, (z - refZ) / scale)
        }

        // ==============================
        // 1. POSE LOOP (132 values)
        // ==============================
        if (pose.isEmpty()) {
            for (v in prevPose) out[k++] = v // ถ้าหาคนไม่เจอ เอาค่าเดิมมาใส่
        } else {
            val startK = k
            for (i in 0 until 33) {
                val targetIdx = swapMap[i] ?: i
                val lm = getOrNullSafe(pose, targetIdx)
                val (rx, ry) = getLandmarkXY(lm)
                val rz = getLandmarkZ(lm)
                val vis = getVis(lm)

                val (fx, fy, fz) = feat(rx, ry, rz)
                out[k++] = fx; out[k++] = fy; out[k++] = fz; out[k++] = vis
            }
            System.arraycopy(out, startK, prevPose, 0, 132) // จำค่าไว้เฟรมหน้า
        }

        // ==============================
        // 2. RIGHT HAND (63 values)
        // ==============================
        val realRightHand = r.rightHandLandmarks()
        if (realRightHand.isEmpty()) {
            // 🔥 FORWARD FILL: เอามือเฟรมที่แล้วมาใส่แทน 0f
            for (v in prevRightHand) out[k++] = v
        } else {
            val startK = k
            for (i in 0 until 21) {
                val lm = getOrNullSafe(realRightHand, i)
                val (rx, ry) = getLandmarkXY(lm)
                val rz = getLandmarkZ(lm)
                val (fx, fy, fz) = feat(rx, ry, rz)
                out[k++] = fx; out[k++] = fy; out[k++] = fz
            }
            System.arraycopy(out, startK, prevRightHand, 0, 63) // จำค่าไว้เฟรมหน้า
        }

        // ==============================
        // 3. LEFT HAND (63 values)
        // ==============================
        val realLeftHand = r.leftHandLandmarks()
        if (realLeftHand.isEmpty()) {
            // 🔥 FORWARD FILL: เอามือเฟรมที่แล้วมาใส่แทน 0f
            for (v in prevLeftHand) out[k++] = v
        } else {
            val startK = k
            for (i in 0 until 21) {
                val lm = getOrNullSafe(realLeftHand, i)
                val (rx, ry) = getLandmarkXY(lm)
                val rz = getLandmarkZ(lm)
                val (fx, fy, fz) = feat(rx, ry, rz)
                out[k++] = fx; out[k++] = fy; out[k++] = fz
            }
            System.arraycopy(out, startK, prevLeftHand, 0, 63) // จำค่าไว้เฟรมหน้า
        }

        return out
    }

    private fun getOrNullSafe(list: List<NormalizedLandmark>, i: Int): NormalizedLandmark? =
        if (i in list.indices) list[i] else null
}
package com.example.hand

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlin.math.hypot

object Holistic258Extractor {

    private const val EPS = 1e-6f

    // ✅ ใช้ false เพื่อให้พิกัด X เป็นลบ (ตรงกับ Python)
    private var mirrorX: Boolean = false

    fun setMirror(mirror: Boolean) {
        mirrorX = false
    }

    fun extract(r: HolisticLandmarkerResult): FloatArray {
        val out = FloatArray(258)
        var k = 0
        val pose = r.poseLandmarks()

        // ฟังก์ชันอ่านค่า (ใช้ logic เดิมที่ค่า X, Y, Z ถูกต้องแล้ว)
        fun getLandmarkXY(lm: NormalizedLandmark?): Pair<Float, Float> {
            if (lm == null) return Pair(0f, 0f)
            val rawX_swapped = lm.y()
            val rawY_swapped = lm.x()
            val finalX = if (mirrorX) 1f - rawX_swapped else rawX_swapped
            return Pair(finalX, rawY_swapped)
        }

        fun getLandmarkZ(lm: NormalizedLandmark?): Float {
            return (lm?.z() ?: 0f) * 2.0f
        }

        fun getVis(lm: NormalizedLandmark?): Float {
            return lm?.visibility()?.orElse(0f) ?: 0f
        }

        // --- คำนวณ Reference จากไหล่จริงก่อนสลับ ---
        val lShoulder = getOrNullSafe(pose, 11)
        val rShoulder = getOrNullSafe(pose, 12)
        val (lxs, lys) = getLandmarkXY(lShoulder)
        val (rxs, rys) = getLandmarkXY(rShoulder)
        val lzs = getLandmarkZ(lShoulder)
        val rzs = getLandmarkZ(rShoulder)

        val ref = if (lShoulder != null && rShoulder != null) {
            val cx = (lxs + rxs) * 0.5f
            val cy = (lys + rys) * 0.5f
            val cz = (lzs + rzs) * 0.5f
            val shoulderW = hypot((lxs - rxs).toDouble(), (lys - rys).toDouble()).toFloat()
            val s = if (shoulderW > EPS) shoulderW else 1f
            Ref(cx, cy, cz, s)
        } else {
            Ref(0f, 0f, 0f, 1f)
        }

        // ==========================================
        // 🔥 ส่วนที่แก้ไข: สลับจุดร่างกาย (Swap Body Parts) 🔥
        // ==========================================
        // เราจะสร้าง map เพื่อสลับซ้ายขวา (เช่น 11<->12, 13<->14)
        // เพื่อให้โมเดลเห็นว่า "แขนซ้ายของคุณ คือ แขนขวาของโมเดล"
        val swapMap = mapOf(
            11 to 12, 12 to 11, // ไหล่
            13 to 14, 14 to 13, // ศอก
            15 to 16, 16 to 15, // ข้อมือ
            17 to 18, 18 to 17, // นิ้วก้อย
            19 to 20, 20 to 19, // นิ้วชี้
            21 to 22, 22 to 21, // นิ้วโป้ง
            23 to 24, 24 to 23, // สะโพก
            25 to 26, 26 to 25, // เข่า
            27 to 28, 28 to 27, // ข้อเท้า
            29 to 30, 30 to 29, // ส้นเท้า
            31 to 32, 32 to 31  // ปลายเท้า
        )

        // Loop Pose (0..32)
        for (i in 0 until 33) {
            // ถ้า i อยู่ใน map ให้ใช้คู่สลับ, ถ้าไม่มีให้ใช้ตัวเดิม (เช่น จมูก 0)
            val targetIdx = swapMap[i] ?: i
            val lm = getOrNullSafe(pose, targetIdx)

            val (rawX, rawY) = getLandmarkXY(lm)
            val rawZ = getLandmarkZ(lm)
            val vis = getVis(lm)

            out[k++] = (rawX - ref.cx) / ref.s
            out[k++] = (rawY - ref.cy) / ref.s
            out[k++] = (rawZ - ref.cz) / ref.s
            out[k++] = vis
        }

        // ==========================================
        // 🔥 ส่วนที่แก้ไข: สลับมือ (Swap Hands) 🔥
        // ==========================================

        // 1. ช่อง Left Hand ของโมเดล -> ให้ใส่ข้อมูลจาก Right Hand ของเรา
        val realRightHand = r.rightHandLandmarks()
        if (realRightHand.isEmpty()) repeat(21 * 3) { out[k++] = 0f }
        else {
            for (i in 0 until 21) {
                val lm = getOrNullSafe(realRightHand, i)
                val (rawX, rawY) = getLandmarkXY(lm)
                val rawZ = getLandmarkZ(lm)
                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s
            }
        }

        // 2. ช่อง Right Hand ของโมเดล -> ให้ใส่ข้อมูลจาก Left Hand ของเรา
        // (นี่คือจุดสำคัญ! ข้อมูลมือซ้ายคุณจะถูกส่งเข้าช่องขวาของโมเดล)
        val realLeftHand = r.leftHandLandmarks()
        if (realLeftHand.isEmpty()) repeat(21 * 3) { out[k++] = 0f }
        else {
            for (i in 0 until 21) {
                val lm = getOrNullSafe(realLeftHand, i)
                val (rawX, rawY) = getLandmarkXY(lm)
                val rawZ = getLandmarkZ(lm)
                out[k++] = (rawX - ref.cx) / ref.s
                out[k++] = (rawY - ref.cy) / ref.s
                out[k++] = (rawZ - ref.cz) / ref.s
            }
        }

        return out
    }

    private fun getOrNullSafe(list: List<NormalizedLandmark>, i: Int): NormalizedLandmark? {
        return if (i in list.indices) list[i] else null
    }

    private data class Ref(val cx: Float, val cy: Float, val cz: Float, val s: Float)
}
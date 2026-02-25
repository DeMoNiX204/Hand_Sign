package com.example.hand

object ModelConfig {

    // =========================
    // ✅ เปลี่ยนโมเดล “แก้ที่นี่ที่เดียว”
    // =========================
    const val MODEL_DIR = "bi-gru-v1.2.1"
    const val TFLITE_NAME = "model_fp16.tflite"
    const val LABELS_NAME = "label_map.json"
    const val THRESH_NAME = "thresholds.json"

    // HolisticLandmarker task (อยู่ root ของ assets)
    const val HOLISTIC_TASK = "holistic_landmarker.task"

    // input shape
    const val SEQ_T = 30
    const val FEAT_F = 258

    // =========================================================
    // ✅ เกณฑ์ “ตอนกดหยุด” ว่าจะตอบหรือไม่ (ปรับ min/max ได้)
    // =========================================================

    // 1) count (จำนวนครั้งที่ผ่านแล้วถูกนับเข้าคลาสนั้น)
    const val MIN_COUNT_TO_ACCEPT = 3
    const val MAX_COUNT_TO_ACCEPT = Int.MAX_VALUE

    // 2) sum (ผลรวมคะแนนของคลาสนั้น)
    const val MIN_SUM_TO_ACCEPT = 0.3f
    const val MAX_SUM_TO_ACCEPT = Float.POSITIVE_INFINITY

    // 3) avg (ค่าเฉลี่ย = sum / count)
    const val MIN_AVG_SCORE_TO_ACCEPT = 0.35f
    const val MAX_AVG_SCORE_TO_ACCEPT = Float.POSITIVE_INFINITY

    // assets paths (Activity จะเรียกใช้อันนี้)
    val MODEL_TFLITE_ASSET: String get() = "$MODEL_DIR/$TFLITE_NAME"
    val MODEL_LABELS_ASSET: String get() = "$MODEL_DIR/$LABELS_NAME"
    val MODEL_THRESH_ASSET: String get() = "$MODEL_DIR/$THRESH_NAME"
}

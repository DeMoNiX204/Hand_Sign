package com.example.hand

import android.content.Context
import com.google.gson.JsonObject
import com.google.gson.JsonParser

data class ThresholdRule(
    val tau: Float,
//    val delta: Float
)

data class ThresholdConfig(
    val tau: Float = 0.5f,
//    val delta: Float = 0.0f,
    val perClass: Map<String, ThresholdRule> = emptyMap()
) {
    fun forLabel(label: String): ThresholdRule = perClass[label] ?: ThresholdRule(tau)

    companion object {
        fun fromAssets(ctx: Context, assetName: String): ThresholdConfig {
            return try {
                val json = ctx.assets.open(assetName).bufferedReader().use { it.readText() }
                val root = JsonParser.parseString(json).asJsonObject
                parseSimple(root)
            } catch (e: Exception) {
                ThresholdConfig() // กรณีไฟล์พังหรือหาไม่เจอ ให้ใช้ค่า Default
            }
        }

        private fun parseSimple(o: JsonObject): ThresholdConfig {
            // 1. ดึงค่า Global (ถ้ามี)
            val gTau = if (o.has("tau")) o.get("tau").asFloat else 0.5f
//            val gDel = if (o.has("delta")) o.get("delta").asFloat else 0.0f

            val per = mutableMapOf<String, ThresholdRule>()

            // 2. วนลูปอ่านรายคลาสตามรูปแบบที่คุณส่งมา
            for ((key, value) in o.entrySet()) {
                // ข้าม key ที่เป็นค่า global
                if (key == "tau") continue

                if (value.isJsonObject) {
                    val obj = value.asJsonObject
                    // อ่านค่า tau/delta ของคลาสนั้นๆ ถ้าไม่มีให้ใช้ค่า Global
                    val t = if (obj.has("tau")) obj.get("tau").asFloat else gTau
                    per[key] = ThresholdRule(t)
                }
            }

            return ThresholdConfig(tau = gTau, perClass = per)
        }
    }
}
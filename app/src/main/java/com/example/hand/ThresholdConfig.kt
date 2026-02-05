package com.example.hand

import android.content.Context
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

data class ThresholdRule(
    val tau: Float,
    val delta: Float
)

data class ThresholdConfig(
    val tau: Float = 0.5f,
    val delta: Float = 0.0f,
    val perClass: Map<String, ThresholdRule> = emptyMap()
) {
    /** คืนค่า threshold สำหรับ label นี้ ถ้าไม่มีใช้ global */
    fun forLabel(label: String): ThresholdRule = perClass[label] ?: ThresholdRule(tau, delta)

    companion object {
        fun fromAssets(ctx: Context, assetName: String): ThresholdConfig {
            val json = ctx.assets.open(assetName).bufferedReader().use { it.readText() }
            val el = JsonParser.parseString(json)

            return when {
                // 0.5
                el.isJsonPrimitive && el.asJsonPrimitive.isNumber ->
                    ThresholdConfig(tau = el.asFloat, delta = 0.0f)

                // [0.5, 0.1]
                el.isJsonArray -> {
                    val a = el.asJsonArray
                    val t = if (a.size() > 0) safeFloat(a[0], 0.5f) else 0.5f
                    val d = if (a.size() > 1) safeFloat(a[1], 0.0f) else 0.0f
                    ThresholdConfig(tau = t, delta = d)
                }

                // object
                el.isJsonObject -> parseRoot(el.asJsonObject)

                else -> ThresholdConfig()
            }
        }

        /**
         * รองรับ:
         * A) global: {"tau":0.5,"delta":0.0}
         * B) per-class (จากเทรนของคุณ): {"fever":{"threshold":0.3,"f1":...}, "no_action":{"threshold":0.4,...}}
         * C) mixed: {"tau":0.5,"delta":0.1,"per_class":{...}} หรือ {"thresholds":{...}}
         */
        private fun parseRoot(o: JsonObject): ThresholdConfig {
            val hasGlobal = o.has("tau") || o.has("delta")
            val gTau = if (hasGlobal) readFloatAny(o.get("tau"), 0.5f) else 0.5f
            val gDel = if (hasGlobal) readFloatAny(o.get("delta"), 0.0f) else 0.0f

            val per = linkedMapOf<String, ThresholdRule>()

            // nested per_class / thresholds
            listOf("per_class", "thresholds").forEach { key ->
                val el = o.get(key)
                if (el != null && el.isJsonObject) {
                    parsePerClass(el.asJsonObject, gTau, gDel, per)
                }
            }

            // pure per-class file: parse every entry except global keys
            for ((k, v) in o.entrySet()) {
                if (k == "tau" || k == "delta" || k == "per_class" || k == "thresholds") continue
                val rule = parseRule(v, gTau, gDel)
                if (rule != null) per[k] = rule
            }

            val outTau = if (hasGlobal) gTau else 0.5f
            val outDel = if (hasGlobal) gDel else 0.0f
            return ThresholdConfig(tau = outTau, delta = outDel, perClass = per)
        }

        private fun parsePerClass(
            pc: JsonObject,
            gTau: Float,
            gDel: Float,
            out: MutableMap<String, ThresholdRule>
        ) {
            for ((label, v) in pc.entrySet()) {
                val rule = parseRule(v, gTau, gDel)
                if (rule != null) out[label] = rule
            }
        }

        /**
         * value แบบที่รับ:
         * - "fever": 0.62
         * - "fever": {"threshold":0.62,"delta":0.08,"f1":...}
         * - รองรับ alias: tau/value และ delta/margin
         */
        private fun parseRule(v: JsonElement, gTau: Float, gDel: Float): ThresholdRule? {
            return try {
                when {
                    v.isJsonPrimitive && v.asJsonPrimitive.isNumber ->
                        ThresholdRule(tau = v.asFloat, delta = gDel)

                    v.isJsonObject -> {
                        val o = v.asJsonObject
                        val t = when {
                            o.has("threshold") -> readFloatAny(o.get("threshold"), gTau)
                            o.has("tau") -> readFloatAny(o.get("tau"), gTau)
                            o.has("value") -> readFloatAny(o.get("value"), gTau)
                            else -> gTau
                        }
                        val d = when {
                            o.has("delta") -> readFloatAny(o.get("delta"), gDel)
                            o.has("margin") -> readFloatAny(o.get("margin"), gDel)
                            else -> gDel
                        }
                        ThresholdRule(tau = t, delta = d)
                    }

                    else -> null
                }
            } catch (_: Throwable) {
                null
            }
        }

        private fun safeFloat(el: JsonElement?, fallback: Float): Float {
            if (el == null) return fallback
            return try {
                if (el.isJsonPrimitive && el.asJsonPrimitive.isNumber) el.asFloat else fallback
            } catch (_: Throwable) {
                fallback
            }
        }

        private fun readFloatAny(el: JsonElement?, fallback: Float): Float {
            if (el == null) return fallback
            return try {
                when {
                    el.isJsonPrimitive && el.asJsonPrimitive.isNumber -> el.asFloat
                    el.isJsonObject && el.asJsonObject.has("value") -> el.asJsonObject.get("value").asFloat
                    else -> fallback
                }
            } catch (_: Throwable) {
                fallback
            }
        }
    }
}

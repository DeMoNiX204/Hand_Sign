package com.example.hand

import android.content.Context
import com.google.gson.JsonParser

data class ThresholdConfig(
    val tau: Float = 0.5f,
    val delta: Float = 0.0f
) {
    companion object {
        fun fromAssets(ctx: Context, assetName: String): ThresholdConfig {
            val json = ctx.assets.open(assetName).bufferedReader().use { it.readText() }
            val el = JsonParser.parseString(json)

            return when {
                // {"tau":0.5,"delta":0.0}
                el.isJsonObject -> {
                    val o = el.asJsonObject
                    ThresholdConfig(
                        tau = readFloat(o.get("tau"), 0.5f),
                        delta = readFloat(o.get("delta"), 0.0f)
                    )
                }

                // 0.5
                el.isJsonPrimitive && el.asJsonPrimitive.isNumber -> {
                    ThresholdConfig(tau = el.asFloat, delta = 0.0f)
                }

                // [0.5, 0.1]
                el.isJsonArray -> {
                    val a = el.asJsonArray
                    val t = if (a.size() > 0) a[0].asFloat else 0.5f
                    val d = if (a.size() > 1) a[1].asFloat else 0.0f
                    ThresholdConfig(tau = t, delta = d)
                }

                else -> ThresholdConfig()
            }
        }

        private fun readFloat(anyEl: com.google.gson.JsonElement?, fallback: Float): Float {
            if (anyEl == null) return fallback
            return try {
                when {
                    anyEl.isJsonPrimitive && anyEl.asJsonPrimitive.isNumber -> anyEl.asFloat
                    // เผื่อบางคนทำ {"tau":{"value":0.5}}
                    anyEl.isJsonObject && anyEl.asJsonObject.has("value") ->
                        anyEl.asJsonObject.get("value").asFloat
                    else -> fallback
                }
            } catch (_: Throwable) {
                fallback
            }
        }
    }
}

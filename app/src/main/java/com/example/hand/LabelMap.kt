package com.example.hand

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class LabelMap private constructor(private val map: Map<Int, String>) {
    val size: Int get() = map.size
    operator fun get(idx: Int): String = map[idx] ?: "class_$idx"

    companion object {
        fun fromAssets(ctx: Context, assetPath: String): LabelMap {
            val json = ctx.assets.open(assetPath).bufferedReader().use { it.readText() }
            val type = object : TypeToken<Map<String, String>>() {}.type
            val raw: Map<String, String> = Gson().fromJson(json, type)
            val m = raw.mapKeys { it.key.toInt() }
            return LabelMap(m)
        }
    }
}

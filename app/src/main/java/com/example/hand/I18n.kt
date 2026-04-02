package com.example.hand

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class I18n private constructor(
    private val dict: Map<String, String>
) {
    fun t(key: String): String = dict[key] ?: key

    companion object {

        fun fromAssets(ctx: Context, lang: String): I18n {
            val path = "i18n/$lang.json"
            val json = ctx.assets.open(path).bufferedReader().use { it.readText() }
            val type = object : TypeToken<Map<String, String>>() {}.type
            val map: Map<String, String> = Gson().fromJson(json, type)
            return I18n(map)
        }
    }
}

package com.example.hand

data class AggBest(
    val idx: Int,
    val count: Int,
    val sum: Float,
    val avg: Float
)

class PredictionAggregator(private val numClasses: Int) {

    private val counts = IntArray(numClasses)
    private val sums = FloatArray(numClasses)

    fun reset() {
        for (i in 0 until numClasses) {
            counts[i] = 0
            sums[i] = 0f
        }
    }

    fun add(idx: Int, score: Float) {
        if (idx !in 0 until numClasses) return
        counts[idx] += 1
        sums[idx] += score
    }

    // เดิม: เลือก best จากทุกคลาส
    fun bestResult(
        minCount: Int,
        maxCount: Int,
        minSum: Float,
        maxSum: Float,
        minAvg: Float,
        maxAvg: Float
    ): AggBest? {
        return bestResultExclude(
            excludeIdx = null,
            minCount = minCount,
            maxCount = maxCount,
            minSum = minSum,
            maxSum = maxSum,
            minAvg = minAvg,
            maxAvg = maxAvg
        )
    }

    // ✅ ใหม่: แบบ A — ไม่ให้บางคลาสชนะ (เช่น no_action)
    fun bestResultExclude(
        excludeIdx: Int?,
        minCount: Int,
        maxCount: Int,
        minSum: Float,
        maxSum: Float,
        minAvg: Float,
        maxAvg: Float
    ): AggBest? {

        var best: AggBest? = null

        for (i in 0 until numClasses) {
            if (excludeIdx != null && i == excludeIdx) continue

            val c = counts[i]
            if (c <= 0) continue

            val s = sums[i]
            val avg = s / c.toFloat()

            // ✅ กรองก่อนเลือก
            if (c < minCount || c > maxCount) continue
            if (s < minSum || s > maxSum) continue
            if (avg < minAvg || avg > maxAvg) continue

            if (best == null || c > best!!.count || (c == best!!.count && s > best!!.sum)) {
                best = AggBest(i, c, s, avg)
            }
        }

        return best
    }
}

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

    fun bestResultExclude(
        excludeIdx: Int?,
        minCount: Int,
        minSum: Float,
        minAvg: Float,
    ): AggBest? {

        var best: AggBest? = null

        for (i in 0 until numClasses) {
            if ( i == excludeIdx) continue

            val c = counts[i]
            if (c <= 0) continue

            val s = sums[i]
            val avg = s / c.toFloat()

            if (c < minCount) continue
            if (s < minSum) continue
            if (avg < minAvg) continue

            if (best == null || c > best!!.count || (c == best!!.count && avg > best!!.avg)) {
                best = AggBest(i, c, s, avg)
            }
        }

        return best
    }
}

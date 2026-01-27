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

    /**
     * เลือกคลาสที่ดีที่สุดแบบ:
     * 1) count มากสุด
     * 2) tie-break ด้วย sum มากสุด
     * แล้วค่อยกรองด้วย min/max ของ count/sum/avg
     */
    fun bestResult(
        minCount: Int,
        maxCount: Int,
        minSum: Float,
        maxSum: Float,
        minAvg: Float,
        maxAvg: Float
    ): AggBest? {

        var bestIdx = -1
        var bestCount = -1
        var bestSum = Float.NEGATIVE_INFINITY

        // เลือกผู้ชนะก่อน
        for (i in 0 until numClasses) {
            val c = counts[i]
            if (c <= 0) continue
            val s = sums[i]

            if (c > bestCount || (c == bestCount && s > bestSum)) {
                bestIdx = i
                bestCount = c
                bestSum = s
            }
        }

        if (bestIdx < 0) return null

        val avg = bestSum / bestCount.toFloat()

        // ✅ กรองด้วยช่วง MIN/MAX
        if (bestCount < minCount) return null
        if (bestCount > maxCount) return null

        if (bestSum < minSum) return null
        if (bestSum > maxSum) return null

        if (avg < minAvg) return null
        if (avg > maxAvg) return null

        return AggBest(bestIdx, bestCount, bestSum, avg)
    }
}

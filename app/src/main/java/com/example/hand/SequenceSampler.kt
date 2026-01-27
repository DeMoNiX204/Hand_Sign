package com.example.hand

object SequenceSampler {

    fun sampleTo30(
        frames: List<FloatArray>,
        seqLen: Int = 30,
        feat: Int = 258
    ): Array<FloatArray> {

        if (frames.isEmpty()) return Array(seqLen) { FloatArray(feat) }

        if (frames.size >= seqLen) {
            val n = frames.size
            val out = Array(seqLen) { FloatArray(feat) }
            for (i in 0 until seqLen) {
                val idx = (i.toDouble() * (n - 1).toDouble() / (seqLen - 1).toDouble()).toInt()
                out[i] = frames[idx]
            }
            return out
        }

        val out = Array(seqLen) { FloatArray(feat) }
        for (i in frames.indices) out[i] = frames[i]
        return out
    }

    fun flatten(seq: Array<FloatArray>, feat: Int = 258): FloatArray {
        val out = FloatArray(seq.size * feat)
        var k = 0
        for (t in seq.indices) {
            System.arraycopy(seq[t], 0, out, k, feat)
            k += feat
        }
        return out
    }
}

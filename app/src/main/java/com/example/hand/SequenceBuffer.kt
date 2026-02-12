package com.example.hand

class SequenceBuffer(private val seqLen: Int, private val featDim: Int) {
    private val frames = Array(seqLen) { FloatArray(featDim) }
    private var count = 0
    private var writeIdx = 0

    fun reset() {
        count = 0
        writeIdx = 0
    }

    fun add(frame: FloatArray) {
        require(frame.size == featDim)
        frame.copyInto(frames[writeIdx])
        writeIdx = (writeIdx + 1) % seqLen
        if (count < seqLen) count++
    }

    fun isFull(): Boolean = count >= seqLen

    fun toFlatFloatArray(): FloatArray {
        val out = FloatArray(seqLen * featDim)
        val start = if (count < seqLen) 0 else writeIdx // oldest
        var k = 0
        for (i in 0 until seqLen) {
            val idx = (start + i) % seqLen
            val f = frames[idx]
            for (j in 0 until featDim) out[k++] = f[j]
        }
        return out
    }
}

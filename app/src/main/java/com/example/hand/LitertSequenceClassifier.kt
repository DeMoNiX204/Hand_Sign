package com.example.hand

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class LitertSequenceClassifier(
    private val context: Context,
    private val modelAssetName: String,
    private val numClasses: Int,
    numThreads: Int = 4
) : AutoCloseable {

    private val interpreter: Interpreter

    private val inputFloats = 30 * 258
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputFloats * 4).order(ByteOrder.nativeOrder())

    private val outputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(numClasses * 4).order(ByteOrder.nativeOrder())

    init {
        val model = loadMappedModel(context, modelAssetName)
        val options = Interpreter.Options().apply { setNumThreads(numThreads) }
        interpreter = Interpreter(model, options)
    }

    fun predict(flat30x258: FloatArray): FloatArray {
        require(flat30x258.size == inputFloats) {
            "Expected $inputFloats floats (30*258), got ${flat30x258.size}"
        }

        inputBuffer.clear()
        for (v in flat30x258) inputBuffer.putFloat(v)
        inputBuffer.rewind()

        outputBuffer.clear()
        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val out = FloatArray(numClasses)
        for (i in 0 until numClasses) out[i] = outputBuffer.getFloat()
        return out
    }

    override fun close() {
        interpreter.close()
    }

    private fun loadMappedModel(ctx: Context, assetName: String): MappedByteBuffer {
        val afd = ctx.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).channel.use { channel ->
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }
}

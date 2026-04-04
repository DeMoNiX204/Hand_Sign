package com.example.hand

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageButton
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        val root = findViewById<View>(R.id.main)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val sys = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            // ไม่ใส่ padding ด้านบน เพื่อให้สีแดงขึ้นไปเต็มด้านบน
            v.setPadding(sys.left, 0, sys.right, sys.bottom)
            insets
        }

        // Buttons
        val fabCamera = findViewById<FrameLayout>(R.id.fabCamera)
//        val btnDebug  = findViewById<ImageButton>(R.id.btnDebug)

        // FAB Camera - เปิดกล้อง
        fabCamera.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        // Debug Button - เปิด Debug Camera
//        btnDebug.setOnClickListener {
//            startActivity(Intent(this, DebugCameraActivity::class.java))
//        }
    }
}
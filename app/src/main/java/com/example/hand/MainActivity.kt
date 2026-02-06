package com.example.hand

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
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

        // Buttons from Design 4
        val fabCamera = findViewById<FrameLayout>(R.id.fabCamera)
        val btnDebug  = findViewById<ImageButton>(R.id.btnDebug)

        // Cards from Design 4
        val cardGuide    = findViewById<CardView>(R.id.cardGuide)
        val cardHistory  = findViewById<CardView>(R.id.cardHistory)
        val cardSettings = findViewById<CardView>(R.id.cardSettings)
        val cardHelp     = findViewById<CardView>(R.id.cardHelp)

        // FAB Camera - เปิดกล้อง
        fabCamera.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        // Debug Button - เปิด Debug Camera
        btnDebug.setOnClickListener {
            startActivity(Intent(this, DebugCameraActivity::class.java))
        }

        // Card: คู่มือใช้งาน
        cardGuide.setOnClickListener {
            Toast.makeText(this, "คู่มือ (ยังไม่ได้ทำหน้า Guide)", Toast.LENGTH_SHORT).show()
        }

        // Card: ประวัติการใช้
        cardHistory.setOnClickListener {
            Toast.makeText(this, "คลังวิดีโอ", Toast.LENGTH_SHORT).show()
        }

        // Card: ตั้งค่า
        cardSettings.setOnClickListener {
            Toast.makeText(this, "เกี่ยวกับApp", Toast.LENGTH_SHORT).show()
        }

        // Card: ช่วยเหลือ
        cardHelp.setOnClickListener {
            Toast.makeText(this, "ผู้จัดทำ", Toast.LENGTH_SHORT).show()
        }
    }
}
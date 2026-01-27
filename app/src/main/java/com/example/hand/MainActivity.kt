package com.example.hand

import android.content.Intent
import android.os.Bundle
import android.view.View
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
            v.setPadding(sys.left, sys.top, sys.right, sys.bottom)
            insets
        }

        val btnCamera = findViewById<ImageButton>(R.id.btnCamera)
        val btnDebug  = findViewById<ImageButton>(R.id.btnDebug)
        val btnHome   = findViewById<ImageButton>(R.id.btnHome)
        val btnHelp   = findViewById<ImageButton>(R.id.btnHelp)
        val cardGuide = findViewById<CardView>(R.id.cardGuide)

        btnCamera.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        btnDebug.setOnClickListener {
            startActivity(Intent(this, DebugCameraActivity::class.java))
        }

        btnHome.setOnClickListener { /* อยู่หน้า Home */ }

        btnHelp.setOnClickListener {
            Toast.makeText(this, "Help", Toast.LENGTH_SHORT).show()
        }

        cardGuide.setOnClickListener {
            Toast.makeText(this, "คู่มือ (ยังไม่ได้ทำหน้า Guide)", Toast.LENGTH_SHORT).show()
        }
    }
}

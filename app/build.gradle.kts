plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

/**
 * ✅ Protobuf fix (สำหรับ MediaPipe Tasks)
 * ใช้ protobuf-java (ตัวเต็ม) เวอร์ชันเดียวทั้งโปรเจกต์
 * และกัน protobuf-javalite ไม่ให้เข้ามาชน
 */
configurations.configureEach {
    exclude(group = "com.google.protobuf", module = "protobuf-javalite")
    resolutionStrategy.force("com.google.protobuf:protobuf-java:3.25.3")
}

android {
    namespace = "com.example.hand"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.hand"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures { viewBinding = true }

    androidResources {
        noCompress += setOf("tflite", "task", "json")
    }

    packaging {
        jniLibs {
            pickFirsts += setOf("**/libc++_shared.so")
        }
        resources {
            excludes += setOf("META-INF/*")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // AndroidX / UI
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)

    // CameraX
    val cameraxVersion = "1.5.0"
    implementation("androidx.cardview:cardview:1.0.0")
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // MediaPipe Tasks Vision (HolisticLandmarker)
    implementation("com.google.mediapipe:tasks-vision:0.10.26.1")

    // LiteRT (Interpreter API) สำหรับรัน model_fp32_v5.tflite
    implementation("com.google.ai.edge.litert:litert:1.4.1")

    // ✅ บังคับ protobuf-java ตัวเต็ม (ให้ชัดไปเลย)
    implementation("com.google.protobuf:protobuf-java:3.25.3")

    // JSON helper (optional)
    implementation("com.google.code.gson:gson:2.11.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

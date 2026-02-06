# Hand App - Design 4: Floating Action

## 📁 ไฟล์ที่ได้

### Layout
- `activity_main_design4.xml` - Layout หลัก

### Drawable Resources (วางใน res/drawable/)
- `bg_top_section_gradient.xml` - Gradient สีแดงสำหรับส่วนบน (มุมล่างโค้งมน)
- `bg_debug_button.xml` - พื้นหลังโปร่งใสสำหรับปุ่ม debug
- `bg_action_icon_gradient.xml` - Gradient สำหรับไอคอนในการ์ด
- `bg_fab_gradient.xml` - Gradient สำหรับปุ่ม FAB (Floating Action Button)

### Styles (เพิ่มใน res/values/styles.xml)
- `styles_fab.xml` - Style สำหรับทำให้ FAB เป็นวงกลมสมบูรณ์

## 🎨 คุณสมบัติ Design 4

### ส่วนบน (Top Section)
- พื้นหลัง gradient สีแดง โค้งมนด้านล่าง
- Title "Hand App" + Subtitle "ยินดีต้อนรับ"
- ปุ่ม Debug มุมบนขวา
- **Stats Cards 2 ใบ** แสดงข้อมูลสรุป:
  - สแกนวันนี้: 12
  - ความแม่นยำ: 95%

### ส่วนกลาง (Content Area)
- **Quick Action Grid 2x2** มี 4 เมนูหลัก:
  1. คู่มือใช้งาน
  2. ประวัติการใช้
  3. ตั้งค่า
  4. ช่วยเหลือ
- แต่ละการ์ดมีไอคอน gradient สีแดงสวยๆ

### Floating Action Button (FAB)
- ปุ่มกล้องลอยมุมล่างขวา
- พื้นหลัง gradient สีแดง
- เงาสวยงาม
- ขนาด 70dp

## 🔧 วิธีติดตั้ง

1. **Copy Layout:**
   ```
   activity_main_design4.xml → res/layout/activity_main.xml
   ```

2. **Copy Drawables:**
   ```
   bg_top_section_gradient.xml → res/drawable/
   bg_debug_button.xml → res/drawable/
   bg_action_icon_gradient.xml → res/drawable/
   bg_fab_gradient.xml → res/drawable/
   ```

3. **เพิ่ม Style:**
   เปิด `res/values/styles.xml` แล้วเพิ่ม:
   ```xml
   <style name="ShapeAppearance.App.CircleFAB" parent="">
       <item name="cornerFamily">rounded</item>
       <item name="cornerSize">50%</item>
   </style>
   ```

4. **เพิ่ม Dependencies (ถ้ายังไม่มี):**
   ใน `build.gradle` (app level):
   ```gradle
   dependencies {
       implementation 'com.google.android.material:material:1.9.0'
       implementation 'androidx.coordinatorlayout:coordinatorlayout:1.2.0'
   }
   ```

## ✨ Customization

### เปลี่ยนข้อความใน Stats Cards:
```xml
<!-- Card 1 -->
android:text="สแกนวันนี้"  // เปลี่ยนหัวข้อ
android:text="12"           // เปลี่ยนตัวเลข

<!-- Card 2 -->
android:text="ความแม่นยำ"  // เปลี่ยนหัวข้อ
android:text="95%"          // เปลี่ยนตัวเลข
```

### เปลี่ยนสีใน Gradient:
แก้ไขใน `bg_top_section_gradient.xml`:
```xml
android:startColor="#E31837"  // สีบน
android:endColor="#C41230"    // สีล่าง
```

### ปรับขนาด FAB:
```xml
app:fabCustomSize="70dp"      // ขนาดปุ่ม
app:maxImageSize="32dp"       // ขนาดไอคอน
```

## 🎯 จุดเด่น
✅ ดูทันสมัย มีความลึก  
✅ Stats cards แสดงข้อมูลสรุปได้ชัดเจน  
✅ Quick actions เข้าถึงง่าย  
✅ FAB ลอยสวยงาม ไม่บัง content  
✅ Scroll ได้ลื่นไหล  

Happy Coding! 🚀🔴⚪

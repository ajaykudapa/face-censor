import cv2
import numpy as np
import subprocess

# === Toggle: Use dynamic or fixed blur ===
use_dynamic_blur = True  # Set to False for fixed blur

# Load face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Input video file
input_path = "sample.mp4"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Output video setup (no audio yet)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("blurred_output.mp4", fourcc, fps, (w, h))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

blur_type = "Dynamic" if use_dynamic_blur else "Fixed"
print(f"🎭 Blurring faces using {blur_type} blur...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1
    percent = (current_frame / frame_count) * 100
    progress = int((current_frame / frame_count) * 40)
    bar = "[" + "#" * progress + "-" * (40 - progress) + "]"
    print(f"\r🎬 {bar} {percent:.1f}% ({current_frame}/{frame_count})", end="")

    # Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_region = frame[y1:y2, x1:x2]

            # === Blur size logic ===
            if use_dynamic_blur:
                face_width = x2 - x1
                face_height = y2 - y1
                blur_size = max(15, int(max(face_width, face_height) / 3))
                if blur_size % 2 == 0:
                    blur_size += 1
            else:
                blur_size = 99  # fixed kernel

            # Apply blur
            blurred_face = cv2.GaussianBlur(face_region, (blur_size, blur_size), 0)

            # Masked elliptical blur
            mask = np.zeros_like(face_region, dtype=np.uint8)
            center = ((x2 - x1) // 2, (y2 - y1) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

            # Blend into frame
            frame[y1:y2, x1:x2] = np.where(mask == 255, blurred_face, face_region)

    out.write(frame)

cap.release()
out.release()

print(f"\n✅ Blurring complete using {blur_type} blur. Now adding audio...")

# Merge original audio
final_output = "blurred_output_with_audio.mp4"
subprocess.run([
    "ffmpeg",
    "-i", "blurred_output.mp4",
    "-i", input_path,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c", "copy",
    "-y",
    final_output
])

print(f"✅ Done! Saved as: {final_output}")
subprocess.run(["open", final_output])

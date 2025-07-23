import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import os

# 🔧 Set tesseract path (change as per your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔁 Load or create the Excel file
EXCEL_FILE = "number_plates.xlsx"
if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
else:
    df = pd.DataFrame(columns=["Number Plate", "Timestamp"])

# 🎯 Define function to detect and read number plate
def recognize_plate(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(cnt)

        # Filter based on area and aspect ratio
        if area > 1000 and 2 < w/h < 6:
            plate_img = frame[y:y+h, x:x+w]
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8 --oem 3')

            # Clean and validate
            plate_text = ''.join(filter(str.isalnum, plate_text))
            if 6 < len(plate_text) < 15:
                return plate_text
    return None

# 🎥 Open webcam
cap = cv2.VideoCapture(0)

print("🚗 Starting Number Plate Recognition... Press 'q' to quit.")

seen_plates = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    plate = recognize_plate(frame)

    if plate and plate not in seen_plates:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Plate Detected: {plate}")

        # Save to Excel DataFrame
        df = pd.concat([df, pd.DataFrame([[plate, timestamp]], columns=["Number Plate", "Timestamp"])], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)

        seen_plates.add(plate)

    # Show video feed
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Exiting and saving data.")

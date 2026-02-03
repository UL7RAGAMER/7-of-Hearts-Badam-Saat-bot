from ultralytics import YOLO
import cv2
import os

# CONFIG
MODEL_PATH = "best2.pt"           # Ensure this matches your file name
TEST_IMAGE = "uploads/images.jpg"  # Your phone image

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return
    
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"Reading {TEST_IMAGE}...")
    original_frame = cv2.imread(TEST_IMAGE)
    if original_frame is None:
        print("❌ Error: Image not found!")
        return

    # --- FIX 1: AUTO-ROTATION LOOP ---
    # Phone images often have wrong rotation metadata. We try 0° and 90°.
    rotations = [
        (None, "Original"),
        (cv2.ROTATE_90_CLOCKWISE, "Rotated 90")
    ]

    best_count = 0
    best_image = None
    best_labels = []

    print("Running Scan...")

    for rotate_code, name in rotations:
        if rotate_code:
            frame = cv2.rotate(original_frame, rotate_code)
        else:
            frame = original_frame

        # --- FIX 2: HIGH RES + STRICTER SETTINGS ---
        results = model(
            frame, 
            verbose=True,            
            # STRICTER: Stops the "hallucinations" of faint/wrong cards
            conf=0.25,        
            
            # OVERLAP: Allows cards to touch/fan out
            iou=0.9,         
            agnostic_nms=True,
        )

        # --- FIX 3: DUPLICATE FILTERING ---
        # Logic: If we see "10h" twice, only keep the one with higher confidence.
        unique_cards = {} # format: {'10h': 0.95, 'ks': 0.88}
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Only store if it's the first time seeing this card OR if this detection is more confident
            if label not in unique_cards or conf > unique_cards[label]:
                unique_cards[label] = conf

        found_count = len(unique_cards)
        print(f"Checked {name}: Found {found_count} unique cards.")

        # Save this result if it's the best one so far
        if found_count > best_count:
            best_count = found_count
            best_image = results[0].plot() # Draws boxes
            best_labels = list(unique_cards.keys())

    # --- FINAL REPORT ---
    if best_count > 0:
        print(f"\n🎉 SUCCESS! Found {best_count} Unique Cards:")
        print(f"Cards: {best_labels}")
        
        cv2.imwrite("final_proof.jpg", best_image)
        print("Saved proof to 'final_proof.jpg'.")
    else:
        print("\n❌ FAILED. No cards detected.")
        print("Tip: If the image is blurry, try taking a photo from further away.")

if __name__ == "__main__":
    main()
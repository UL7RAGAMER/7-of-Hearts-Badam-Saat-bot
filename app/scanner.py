import cv2
from ultralytics import YOLO

# Mapping YOLO labels (standard) to your App's Suits/Ranks
# Adjust these keys based on the specific model you download!
YOLO_SUIT_MAP = {'h': 0, 'd': 1, 'c': 2, 's': 3} # Hearts, Diamonds, Clubs, Spades
YOLO_RANK_MAP = {
    'a': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, 
    '7': 6, '8': 7, '9': 8, '10': 9, 'j': 10, 'q': 11, 'k': 12
}   

class CardScanner:
    def __init__(self, model_path='best2.pt'):
        try:
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            print(f"Scanner Warning: Could not load model ({e})")
            self.available = False

    def capture_and_detect(self):
        """
        Opens webcam. Press 'SPACE' to capture and detect. Press 'Q' to quit.
        Returns: Set of card IDs (0-51) detected.
        """
        if not self.available:
            print("Model not loaded.")
            return set()

        cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        detected_ids = set()
        
        print("Camera Open: Press SPACE to Scan, Q to Quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Show the camera feed
            cv2.imshow("Badaam Sath Scanner (Space=Scan, Q=Quit)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'SPACE' to scan the current frame
            if key == 32: 
                print("Scanning frame...")
                results = self.model(frame, verbose=False)
                
                detected_ids.clear()
                for r in results:
                    for box in r.boxes:
                        # Get label (e.g., "10h", "Kd")
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        
                        # Convert label to ID
                        card_id = self._label_to_id(label)
                        if card_id is not None:
                            detected_ids.add(card_id)
                
                print(f"Detected {len(detected_ids)} cards: {detected_ids}")
                # Flash effect or simple console confirmation
                
            # Press 'Q' to close and return results
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return detected_ids



    def detect_from_file(self, image_path):
        if not self.available: return set()

        original_frame = cv2.imread(image_path)
        if original_frame is None: return set()

        # Try both orientations (standard and 90-degree rotation for mobile)
        rotations = [
            (None, "Original"),
            (cv2.ROTATE_90_CLOCKWISE, "Rotated 90")
        ]

        best_detected_ids = set()

        for rotate_code, name in rotations:
            frame = cv2.rotate(original_frame, rotate_code) if rotate_code else original_frame
            
            # High-res inference with stricter settings from test1.py
            results = self.model(frame, verbose=False, conf=0.25, iou=0.90, agnostic_nms=True)
            
            # Filter duplicates: Keep highest confidence for each unique label
            unique_cards_in_rotation = {} 
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                if label not in unique_cards_in_rotation or conf > unique_cards_in_rotation[label]:
                    unique_cards_in_rotation[label] = conf

            # Convert labels to IDs
            current_ids = set()
            for label in unique_cards_in_rotation.keys():
                cid = self._label_to_id(label)
                if cid is not None:
                    current_ids.add(cid)
            
            # If this rotation found more cards, use it
            if len(current_ids) > len(best_detected_ids):
                best_detected_ids = current_ids

        return best_detected_ids

    def _label_to_id(self, label):
        label = label.lower()
        suit_code = label[-1]
        rank_code = label[:-1]
        
        if suit_code in YOLO_SUIT_MAP and rank_code in YOLO_RANK_MAP:
            return YOLO_SUIT_MAP[suit_code] * 13 + YOLO_RANK_MAP[rank_code]
        return None
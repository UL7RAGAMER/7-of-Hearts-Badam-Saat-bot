from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load the Fast Nano Model
    model = YOLO('best.pt') 

    # 2. Train with "Real World" Augmentations
    results = model.train(
        data='datasets/playing_cards/data.yaml', 
        epochs=200,
        imgsz=640,
        batch=16*6, 
        cache=True,
        workers=0,
        name='long_new',
        device=0,
        
        # --- THE KEY FIXES ---
        degrees=60.0,      # ALLOW ROTATION: Randomly rotate images +/- 60 degrees
        shear=15.0,        # PERSPECTIVE: Simulate looking at cards from an angle
        scale=0.6,         # ZOOM: Randomly zoom in/out (teaches small/partial cards)
        mosaic=1.0,        # MOSAIC: Mashes 4 images together (teaches edges)
        erasing=0.4,       # OCCLUSION: Randomly erase black boxes on cards (teaches hidden parts)
        mixup=0.35,        # TRANSPARENCY: Blends 2 images (teaches overlap)
    )

    print("Training Complete.")

if __name__ == '__main__':
    main()
import os
import cv2
import torch
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline

#*************************************************************************************
# Capture a centered square image from webcam and save as 512x512
#*************************************************************************************

def capture_square_image(target_size=512, cam_id=1, out_path="opencv_frame.png"):
    print("[INFO] Starting webcam capture...")
    cam = cv2.VideoCapture(cam_id)
    cv2.namedWindow("Square Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Square Capture", target_size, target_size)
    frame_square = None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        h, w, _ = frame.shape
        side = min(h, w)
        start_x = (w - side) // 2
        start_y = (h - side) // 2
        frame_square = frame[start_y:start_y+side, start_x:start_x+side]
        frame_square = cv2.resize(frame_square, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        cv2.imshow("Square Capture", frame_square)

        k = cv2.waitKey(1)
        if k%256 == 27:
            print("[INFO] ESC pressed, closing webcam window...")
            frame_square = None
            break
        elif k%256 == 32:
            cv2.imwrite(out_path, frame_square)
            print(f"[INFO] Image captured and saved as {out_path}!")
            break

    cam.release()
    cv2.destroyAllWindows()
    return frame_square

#*************************************************************************************
# Use YOLOv8 to detect the person
#*************************************************************************************

def detect_person_yolo(frame):
    # Load YOLOv8 model (nano version for speed, can use 's', 'm', 'l', 'x' for larger models)
    print("[INFO] Detecting person with YOLOv8...")
    yolo_model = YOLO('yolov8n.pt')
    # Run YOLOv8 detection
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    classes = results[0].boxes.cls.cpu().numpy()  # class indices
    # Find the largest 'person' box (class 0 in COCO)
    person_boxes = [box for box, cls in zip(boxes, classes) if int(cls) == 0]
    if not person_boxes:
        print("[ERROR] No person detected!")
        raise Exception("No person detected!")
    person_box = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    return person_box

#*************************************************************************************
# Generate masks with SAM
#*************************************************************************************

def segment_person_sam(frame, person_box):  
    print("[INFO] Using SAM with YOLO bounding box...")
    sam = sam_model_registry["vit_b"](
        checkpoint="./model/sam_vit_b_01ec64.pth",
    ).to('cpu')

    predictor = SamPredictor(sam)
    predictor.set_image(frame)

    input_box = np.array(person_box).astype(int)[None, :]
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=True
    )
    best_mask = masks[np.argmax([m.sum() for m in masks])]
    combined_mask = best_mask
    print("[INFO] Mask generated using SAM and YOLO bounding box.")
    return best_mask

#*************************************************************************************
# Générer un nouveau fond avec Stable Diffusion InpaintPipeline
#*************************************************************************************

def inpaint_background(original_image, mask, prompt, model_path):
    print("[INFO] Generating new background with Stable Diffusion...")
    # Préparer le masque binaire pour l'inpainting (le fond doit être blanc, le sujet noir)
    # Ici, on inverse le masque: le fond (False) devient 255, le sujet (True) devient 0
    mask_pil = Image.fromarray((~mask * 255).astype("uint8"))
    # Redimensionner le masque si besoin (doit avoir la même taille que l'image)
    mask_pil = mask_pil.resize(original_image.size)

    # Charger le pipeline d'inpainting
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=model_path
    ).to("mps")
    # Générer l'image avec le nouveau fond
    with torch.no_grad():
        out_image = inpaint_pipe(
            prompt=prompt,
            image=original_image,
            mask_image=mask_pil,
            strength=0.9,
            generator=torch.Generator("mps").manual_seed(7)
        ).images[0]

    # Sauvegarder et afficher le résultat
    out_image.save("output_inpaint.png")
    print("[INFO] Final image saved as output_inpaint.png.")
    out_image.show()
    print("[INFO] Processing complete!")

def main():
    # Paths
    grand_parent_dir = os.path.dirname(os.getcwd())
    model_path = os.path.join(grand_parent_dir, "model")
    prompt = input("Enter the prompt for the new background: ")
    print(f"[INFO] Using prompt: {prompt}")

    # 1. Capture 
    frame = capture_square_image(target_size=512)
    if frame is None:
        return

    # 2. YOLO
    person_box = detect_person_yolo(frame)

    # 3. SAM
    combined_mask = segment_person_sam(frame, person_box)

    # 4. PNG transparent
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([frame_rgb, combined_mask.astype(np.uint8) * 255])
    Image.fromarray(rgba).save("foreground.png")
    print("[INFO] Transparent PNG saved as foreground.png.")

    # 5. Inpainting
    original_image = Image.fromarray(frame_rgb)
    inpaint_background(original_image, combined_mask, prompt, model_path)

    print("[INFO] Processing complete!")

if __name__ == "__main__":
    main()
"""
Grounding DINO + SAM — Text-prompted auto-segmentation pipeline.

Pipeline:
    Text prompt → Grounding DINO → bounding boxes → SAM → pixel masks

Uses:
  - transformers 4.46+ for Grounding DINO (AutoModelForZeroShotObjectDetection)
  - segment_anything (SAM v1) for mask generation — already installed and compatible with PyTorch 2.3

Usage:
    worker = DinoSamWorker(image_2d, prompts=["void", "crack"], class_mapping={"void": 1})
    worker.finished.connect(on_result)
    worker.start()
"""
import traceback
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

# Lazy imports — heavy libs loaded only when needed
_gdino_model = None
_gdino_processor = None
_sam_predictor = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_gdino():
    """Load Grounding DINO model (cached after first call)."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return _gdino_processor, _gdino_model
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GDINO_ID = "IDEA-Research/grounding-dino-base"
    _gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(_get_device())
    _gdino_model.eval()
    return _gdino_processor, _gdino_model


def _load_sam(checkpoint_path=None):
    """
    Load SAM v1 model (cached after first call).
    
    If checkpoint_path is None, tries to find a cached SAM checkpoint.
    Falls back to vit_b as default model type.
    """
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    
    if checkpoint_path is None:
        # Try common paths
        import os
        candidates = [
            os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth"),
            os.path.expanduser("~/.cache/sam/sam_vit_l_0b3195.pth"),
            os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
            "sam_vit_b_01ec64.pth",
            "sam_vit_l_0b3195.pth",
        ]
        for cp in candidates:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            # Auto-download vit_b (smallest, ~375MB)
            import urllib.request
            os.makedirs(os.path.expanduser("~/.cache/sam"), exist_ok=True)
            checkpoint_path = os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            print(f"Downloading SAM vit_b checkpoint (~375MB)...")
            urllib.request.urlretrieve(url, checkpoint_path)
            print("Download complete.")
    
    # Detect model type from filename
    mt = "vit_b"
    if "vit_l" in checkpoint_path.lower():
        mt = "vit_l"
    elif "vit_h" in checkpoint_path.lower():
        mt = "vit_h"
    
    sam = sam_model_registry[mt](checkpoint=checkpoint_path)
    if torch.cuda.is_available():
        sam.to(device="cuda")
    _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def detect(image_pil, text_prompts, box_threshold=0.25, text_threshold=0.25):
    """
    Run Grounding DINO on a PIL RGB image.
    
    Returns dict: boxes (Tensor N,4), scores (Tensor N), text_labels (list[str])
    """
    import torch
    processor, model = _load_gdino()
    device = _get_device()
    
    # Clean prompts: each item is a separate detection target
    cleaned = [p.lower().strip().rstrip('.') for p in text_prompts]
    # Grounding DINO expects: "prompt1. prompt2. prompt3."
    text_string = ". ".join(cleaned) + "."
    
    print(f"[DINO] Text input: '{text_string}' | box_threshold={box_threshold}")
    
    inputs = processor(images=image_pil, text=text_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    try:
        # Older transformers versions (e.g. 4.46.3) use 'box_threshold'
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )
    except TypeError:
        # Newer transformers versions (>= 4.51.0) renamed it to 'threshold'
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )
    
    r = results[0]
    n_boxes = len(r.get("boxes", []))
    print(f"[DINO] Raw detections: {n_boxes} boxes")
    if n_boxes > 0:
        for i in range(min(n_boxes, 10)):
            lbl = r.get("text_labels", r.get("labels", []))[i] if i < len(r.get("text_labels", r.get("labels", []))) else "?"
            score = r["scores"][i].item()
            box = r["boxes"][i].tolist()
            print(f"  [{i}] label='{lbl}' score={score:.3f} box={[round(v,1) for v in box]}")
    
    return r


def apply_nms(results, iou_threshold=0.5):
    """Remove overlapping duplicate boxes with Non-Maximum Suppression."""
    from torchvision.ops import nms
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results.get("text_labels", results.get("labels", []))
    if len(boxes) == 0:
        return results
    keep = nms(boxes, scores, iou_threshold)
    print(f"[DINO] After NMS (iou={iou_threshold}): {len(keep)}/{len(boxes)} kept")
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "text_labels": [labels[i] for i in keep.tolist()]
    }


def filter_large_boxes(results, image_pil, max_area_fraction=0.7):
    """Remove boxes covering more than max_area_fraction of the image."""
    img_w, img_h = image_pil.size
    img_area = img_w * img_h
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results.get("text_labels", results.get("labels", []))
    keep = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box.tolist()
        area_frac = ((x1 - x0) * (y1 - y0)) / img_area
        if area_frac < max_area_fraction:
            keep.append(i)
        else:
            print(f"[DINO] Filtered large box #{i}: area_frac={area_frac:.2f} > {max_area_fraction}")
    if len(keep) < len(boxes):
        print(f"[DINO] After large-box filter: {len(keep)}/{len(boxes)} kept")
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "text_labels": [labels[i] for i in keep]
    }


def segment_with_sam(image_np_rgb, boxes_tensor, sam_predictor=None):
    """
    Run SAM v1 on an image using bounding boxes from Grounding DINO.
    
    Args:
        image_np_rgb: np.ndarray (H, W, 3) uint8 RGB
        boxes_tensor: torch.Tensor (N, 4) xyxy pixel coordinates
        sam_predictor: optional pre-loaded SamPredictor
    
    Returns:
        masks: np.ndarray (N, H, W) boolean
        iou_scores: list of float
    """
    if len(boxes_tensor) == 0:
        h, w = image_np_rgb.shape[:2]
        return np.zeros((0, h, w), dtype=bool), []
    
    predictor = sam_predictor or _load_sam()
    predictor.set_image(image_np_rgb)
    
    boxes_np = boxes_tensor.cpu().numpy()
    all_masks = []
    all_ious = []
    
    for box in boxes_np:
        masks, iou_scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )
        # Pick best mask (highest IoU)
        best_idx = int(np.argmax(iou_scores))
        all_masks.append(masks[best_idx])
        all_ious.append(float(iou_scores[best_idx]))
    
    return np.array(all_masks), all_ious


def run_pipeline(image_pil, prompts, box_threshold=0.25, nms_iou=0.5, sam_predictor=None):
    """
    Full Grounding DINO + SAM pipeline.
    
    Args:
        image_pil: PIL.Image (RGB)
        prompts: list of str (e.g. ["void", "crack"])
        box_threshold: float
        nms_iou: float
        sam_predictor: optional pre-loaded SamPredictor
    
    Returns:
        results: dict (boxes, scores, text_labels)
        masks: np.ndarray (N, H, W) boolean
        iou_scores: list of float
    """
    results = detect(image_pil, prompts, box_threshold=box_threshold)
    results = filter_large_boxes(results, image_pil)
    results = apply_nms(results, iou_threshold=nms_iou)
    
    if len(results["boxes"]) == 0:
        w, h = image_pil.size
        print(f"[DINO] No detections after filtering — returning empty masks")
        return results, np.zeros((0, h, w), dtype=bool), []
    
    image_rgb = np.array(image_pil)
    masks, iou_scores = segment_with_sam(image_rgb, results["boxes"], sam_predictor)
    print(f"[SAM] Generated {len(masks)} masks, IoU scores: {[round(s,3) for s in iou_scores]}")
    return results, masks, iou_scores


class DinoSam2LoadThread(QThread):
    """Background thread to pre-load both Grounding DINO and SAM models."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, error_msg

    def __init__(self, sam_checkpoint=None):
        super().__init__()
        self.sam_checkpoint = sam_checkpoint

    def run(self):
        try:
            self.progress.emit("Loading Grounding DINO (detection)...")
            _load_gdino()
            self.progress.emit("Loading SAM (segmentation)...")
            _load_sam(self.sam_checkpoint)
            self.progress.emit("✓ Models ready!")
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, f"{str(e)}\n{traceback.format_exc()}")


class DinoSam2Worker(QThread):
    """
    Run the DINO+SAM pipeline on a single 2D slice.
    
    Emits:
        progress(str): status messages
        finished(np.ndarray, dict): combined_mask (H,W uint8), info_dict
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # mask_2d (H,W uint8), info_dict

    def __init__(self, image_2d, prompts, class_mapping=None,
                 box_threshold=0.25, nms_iou=0.5, min_iou_score=0.0,
                 sam_predictor=None):
        """
        Args:
            image_2d: np.ndarray (H,W) grayscale uint8 or (H,W,3) RGB
            prompts: list of str, e.g. ["void", "crack"]
            class_mapping: dict mapping text_label → class_id, e.g. {"void": 1, "crack": 2}
                          If None, all detections get class_id=1
            box_threshold: float, detection confidence threshold
            nms_iou: float, NMS IoU threshold
            min_iou_score: float, minimum SAM IoU score to accept a mask
            sam_predictor: optional pre-loaded SamPredictor
        """
        super().__init__()
        self.image_2d = image_2d
        self.prompts = prompts
        self.class_mapping = class_mapping or {}
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.min_iou_score = min_iou_score
        self.sam_predictor = sam_predictor

    def run(self):
        try:
            from PIL import Image

            # Convert to PIL RGB
            img = self.image_2d
            print(f"[DinoSam2Worker] Image shape={img.shape} dtype={img.dtype} "
                  f"min={img.min()} max={img.max()}")
            
            if img.ndim == 2:
                img_pil = Image.fromarray(img).convert("RGB")
            else:
                img_pil = Image.fromarray(img)

            self.progress.emit("Running Grounding DINO + SAM...")
            results, masks, iou_scores = run_pipeline(
                img_pil, self.prompts,
                box_threshold=self.box_threshold,
                nms_iou=self.nms_iou,
                sam_predictor=self.sam_predictor,
            )

            n_detected = len(results.get("boxes", []))
            if n_detected == 0 or len(masks) == 0:
                self.progress.emit("No objects detected.")
                h, w = self.image_2d.shape[:2]
                self.finished.emit(
                    np.zeros((h, w), dtype=np.uint8),
                    {"n_detected": 0, "labels": [], "scores": [], "iou_scores": []}
                )
                return

            # Compose combined class mask
            h, w = masks.shape[1], masks.shape[2]
            combined = np.zeros((h, w), dtype=np.uint8)
            labels = results.get("text_labels", [])
            scores = results["scores"].cpu().numpy().tolist()

            accepted = 0
            for i in range(n_detected):
                iou = iou_scores[i] if i < len(iou_scores) else 0.0
                if iou < self.min_iou_score:
                    continue
                label = labels[i] if i < len(labels) else ""
                class_id = self.class_mapping.get(label, 1)
                combined[masks[i]] = class_id
                accepted += 1

            info = {
                "n_detected": n_detected,
                "n_accepted": accepted,
                "labels": labels,
                "scores": scores,
                "iou_scores": iou_scores,
            }
            self.progress.emit(f"Done! {accepted}/{n_detected} objects accepted.")
            self.finished.emit(combined, info)

        except Exception as e:
            self.progress.emit(f"Error: {e}")
            print(f"[DinoSam2Worker] ERROR: {e}\n{traceback.format_exc()}")
            h, w = self.image_2d.shape[:2]
            self.finished.emit(
                np.zeros((h, w), dtype=np.uint8),
                {"error": str(e), "traceback": traceback.format_exc()}
            )

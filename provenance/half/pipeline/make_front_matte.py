import numpy as np, cv2, torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from settings import WORK, SAM2_WEIGHTS, input_path

src_path = input_path("01_front.png")
out_dir = WORK / "inputs"
out_dir.mkdir(parents=True, exist_ok=True)
arr = np.array(Image.open(src_path).convert("RGB"))
H, W = arr.shape[:2]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_WEIGHTS), device=device)
predictor = SAM2ImagePredictor(model)
predictor.set_image(arr)

box = np.array([int(W*0.25), int(H*0.015), int(W*0.74), int(H*0.965)], dtype=np.float32)
pos = np.array([[int(W*0.50), int(H*0.28)], [int(W*0.44), int(H*0.08)], [int(W*0.50), int(H*0.42)]], dtype=np.float32)
lbl = np.array([1, 1, 1], dtype=np.int32)
masks, scores, _ = predictor.predict(point_coords=pos, point_labels=lbl, box=box, multimask_output=True)
best = int(np.argmax(scores))
mask = (masks[best] > 0.5).astype(np.uint8) * 255
print("Score:", scores[best])

num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask)
largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
mask = np.where(labels_im == largest, 255, 0).astype(np.uint8)

# Check interior holes and only close genuine tiny pinholes (< 150 px)
inv = 255 - mask
n2, l2, s2, _ = cv2.connectedComponentsWithStats(inv)
for cid in range(1, n2):
    x, y, w, h, area = s2[cid]
    touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)
    if not touches_border:
        print(f"  interior hole cid={cid} bbox=({x},{y},{w},{h}) area={area}")
        if area < 150:
            mask[l2 == cid] = 255

# Also cut out any studio-gray pocket between right waist & right arm (x: 530..610, y: 600..850) where pixel matches neutral gray backdrop
gray_bg = (
    (np.max(arr, axis=2) - np.min(arr, axis=2) <= 10) &
    (arr.mean(axis=2) >= 165) & (arr.mean(axis=2) <= 215)
)
# Only inside that known arm-torso pocket connected to the gap!
pocket = np.zeros((H, W), dtype=np.uint8)
pocket[int(H*0.36):int(H*0.49), int(W*0.54):int(W*0.62)] = gray_bg[int(H*0.36):int(H*0.49), int(W*0.54):int(W*0.62)].astype(np.uint8) * 255
# Keep largest gray pocket component if substantial (>300 px)
npk, lpk, spk, _ = cv2.connectedComponentsWithStats(pocket)
for cid in range(1, npk):
    if spk[cid, cv2.CC_STAT_AREA] > 300:
        print(f"  removed gray arm-waist pocket area={spk[cid, cv2.CC_STAT_AREA]}")
        mask[lpk == cid] = 0

alpha = cv2.GaussianBlur(mask, (3, 3), 0.7)
core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
alpha = np.where(core > 0, 255, alpha)

rgba = np.dstack([arr, alpha])
Image.fromarray(rgba).save(f"{out_dir}/01_front_rgba.png")

bg = np.zeros_like(arr); bg[:] = [255, 0, 200]
a = (alpha.astype(np.float32) / 255.0)[..., None]
qc = (arr * a + bg * (1.0 - a)).astype(np.uint8)
Image.fromarray(qc).save(f"{out_dir}/01_front_cutout_qc.png")

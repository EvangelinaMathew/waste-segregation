import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
VIDEOS = ROOT / "videos"
EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

for cls_dir in sorted(p for p in VIDEOS.iterdir() if p.is_dir()):
    out = ROOT / "scraped" / cls_dir.name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for v in sorted(cls_dir.iterdir()):
        if v.suffix.lower() not in EXTS:
            continue
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(fps / 8))        # about 8 frames per second
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out / f"vid_{v.stem}_{i:05d}.jpg"), frame)
                n += 1
            i += 1
        cap.release()
    print(cls_dir.name, "->", n, "frames")
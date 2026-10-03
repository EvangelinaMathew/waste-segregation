"""
build_dataset.py — replaces preprocess.py
Merges Kaggle data (data/raw) + scraped images (data/scraped)
-> data/processed/train/<class>/*.jpg and data/processed/val/<class>/*.jpg
Run: python backend/build_dataset.py
"""
import hashlib, random, shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW, SCRAPED, OUT = ROOT / "raw", ROOT / "scraped", ROOT / "processed"
SIZE = (224, 224)
MAX_PER_CLASS = 1000         # lower this if training is too slow
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# source folder name (lowercase) -> final class name
FOLDER_MAP = {
    # --- your current Kaggle dataset (phenomsg) ---
    "coffee_tea_bags": "egg_shells_tea", "egg_shells": "egg_shells_tea",
    "food_scraps": "food_waste", "kitchen_waste": "food_waste",
    "yard_trimmings": "yard_waste",
    "batteries": "battery", "e-waste": "ewaste",
    "paints": "paint_chemical", "pesticides": "paint_chemical",
    "cans_all_type": "metal_can", "glass_containers": "glass",
    "paper_products": "paper", "plastic_bottles": "plastic_bottle",
    "ceramic_product": "ceramic", "diapers": "diaper_sanitary",
    "sanitary_napkin": "diaper_sanitary",
    "platics_bags_wrappers": "plastic_bag",   # (sic) typo is in the dataset
    "stroform_product": "thermocol",
    # --- 'Garbage Classification' (mostafaabla) ---
    "battery": "battery", "biological": "food_waste",
    "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass",
    "cardboard": "paper", "paper": "paper", "metal": "metal_can",
    "plastic": "plastic_bottle",
}

files = {}
def add(cls, p): files.setdefault(cls, []).append(p)

# 1) Kaggle datasets: the nearest parent folder that is in FOLDER_MAP decides the class
for p in RAW.rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS:
        for part in reversed(p.relative_to(RAW).parts[:-1]):
            if part.lower() in FOLDER_MAP:
                add(FOLDER_MAP[part.lower()], p)
                break

# 2) Scraped images: folder name IS the class name
if SCRAPED.exists():
    for p in SCRAPED.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            cls = p.relative_to(SCRAPED).parts[0]
            add(cls, p)

random.seed(42)
if OUT.exists():
    shutil.rmtree(OUT)          # processed/ is rebuilt from raw+scraped each run

seen, summary = set(), {}
for cls, paths in sorted(files.items()):
    random.shuffle(paths)
    n = 0
    for p in paths:
        if n >= MAX_PER_CLASS:
            break
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize(SIZE, Image.LANCZOS)
                h = hashlib.md5(im.tobytes()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                split = "val" if n % 7 == 0 else "train"   # ~14% validation
                d = OUT / split / cls
                d.mkdir(parents=True, exist_ok=True)
                im.save(d / f"{n:05d}.jpg", quality=90)
                n += 1
        except Exception:
            continue            # corrupt image, skip
    summary[cls] = n
    print(f"{cls:<16} {n}")

print("\nTotal:", sum(summary.values()), "images,", len(summary), "classes")
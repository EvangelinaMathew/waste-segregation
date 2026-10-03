from icrawler.builtin import BingImageCrawler
import os

ROOT = os.path.expanduser("~/Projects/waste-segregation/data/scraped")

# folder_name: [search phrases]. More phrases = more variety.
QUERIES = {
    "plastic_bottle":  ["plastic bottle waste", "crushed plastic bottle", "empty water bottle trash",
                        "plastic bottle in hand", "water bottle on table"],
    "paper":           ["waste paper pile", "crumpled paper", "used cardboard box"],
    "glass":           ["empty glass bottle waste", "broken glass jar trash"],
    "metal_can":       ["crushed soda can", "empty tin can waste"],
    "food_waste":      ["leftover food waste", "vegetable peels waste", "rotten fruit"],
    "egg_shells_tea":  ["egg shells", "used tea bags", "coffee grounds waste"],
    "yard_waste":      ["dry leaves garden waste", "grass clippings"],
    "battery":         ["used batteries waste", "dead AA batteries"],
    "ewaste":          ["old mobile phone e-waste", "broken electronics waste", "old charger cable"],
    "paint_chemical":  ["paint can waste", "pesticide bottle", "chemical container"],
    "plastic_bag":     ["plastic bag waste", "chips wrapper trash", "food packaging wrapper"],
    "thermocol":       ["thermocol waste", "styrofoam packaging"],
    "diaper_sanitary": ["used diaper waste", "sanitary pad waste"],
    "ceramic":         ["broken ceramic plate", "broken coffee mug"],
    "pen":             ["ballpoint pen", "pen on desk", "person holding pen", "used pen waste", "gel pen"],
    "plastic_cup":     ["disposable plastic cup", "used plastic cup trash", "plastic glass waste"],
    "paper_cup":       ["disposable paper cup", "used coffee cup waste", "paper cup trash"],
    "background":      ["empty desk", "human hand holding nothing", "plain wall", "room floor", "empty table top"],
}

for cls, phrases in QUERIES.items():
    out = os.path.join(ROOT, cls)
    os.makedirs(out, exist_ok=True)
    for p in phrases:
        crawler = BingImageCrawler(storage={"root_dir": out},
                                   downloader_threads=8)
        crawler.crawl(keyword=p, max_num=600, min_size=(200, 200),
                      file_idx_offset="auto")
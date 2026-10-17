from icrawler.builtin import BingImageCrawler, BaiduImageCrawler, GoogleImageCrawler
from pathlib import Path
import logging, time

ROOT = Path.home() / "Projects/waste-segregation/data/scraped"
PER_QUERY = 150          # images requested per phrase
STOP_AT = 1500           # stop adding to a folder beyond this

# folder name (= class name) -> search phrases
QUERIES = {
    "ewaste": [
        "neckband earphones", "bluetooth neckband", "wired earphones", "earbuds charging case",
        "headphones on desk", "phone charger adapter", "usb charging cable", "power bank",
        "smartphone in hand", "old mobile phone", "smartwatch on wrist", "computer mouse",
        "computer keyboard", "tv remote control", "calculator", "pen drive usb",
        "extension cord", "old electronics pile",
    ],
    "battery": [
        "AA battery in hand", "AAA batteries", "button cell battery", "9v battery",
        "rechargeable batteries", "dead batteries", "battery on table",
    ],
    "pen": [
        "ballpoint pen in hand", "pen on desk", "gel pen", "marker pen", "highlighter pen",
        "pen on notebook", "blue pen close up", "used pen",
    ],
    "pencil": [
        "pencil in hand", "pencil on desk", "wooden pencil", "sharpened pencil",
        "pencils on table", "pencil and eraser", "mechanical pencil", "colour pencils",
    ],
    "plastic_bottle": [
        "plastic water bottle in hand", "water bottle on desk", "cold drink bottle",
        "empty pet bottle", "plastic bottle on table", "mineral water bottle",
        "juice bottle plastic", "squeezed plastic bottle",
    ],
    "metal_can": [
        "steel water bottle in hand", "metal flask", "aluminium can in hand",
        "soda can on table", "tin can", "thermos bottle", "crushed beer can",
    ],
    "glass": [
        "glass bottle in hand", "empty glass jar", "glass bottle on table",
        "broken glass bottle", "wine bottle empty", "glass tumbler",
    ],
    "paper": [
        "notebook on desk", "newspaper pile", "cardboard box", "paper sheets on table",
        "crumpled paper ball", "book on table", "paper bag", "printed documents",
    ],
    "carton": [
        "tetra pak juice box", "milk carton", "juice carton", "cardboard food packaging",
        "pizza box", "egg carton",
    ],
    "plastic_container": [
        "plastic food container", "lunch box plastic", "plastic tub", "yogurt cup plastic",
        "plastic jar", "takeaway container", "plastic cup in hand",
    ],
    "plastic_cutlery": [
        "plastic spoon", "disposable plastic fork", "plastic straw", "disposable cutlery",
        "plastic plate disposable",
    ],
    "plastic_bag": [
        "plastic carry bag", "chips packet empty", "snack wrapper", "biscuit wrapper",
        "plastic bag in hand", "chocolate wrapper",
    ],
    "paper_cup": [
        "paper coffee cup in hand", "disposable paper cup", "tea paper cup",
        "used coffee cup", "paper cup on table",
    ],
    "food_waste": [
        "banana peel", "fruit peels waste", "vegetable scraps", "leftover food plate",
        "apple core", "orange peel", "rotten vegetables", "cooked rice leftover",
    ],
    "egg_shells_tea": ["egg shells in hand", "used tea bag", "coffee grounds", "tea leaves waste"],
    "ceramic": [
        "ceramic cup on table", "broken ceramic plate", "porcelain mug", "ceramic bowl",
        "broken coffee mug", "ceramic tile pieces",
    ],
    "thermocol": ["thermocol sheet", "styrofoam packaging", "thermocol box", "styrofoam cup"],
    "diaper_sanitary": ["used diaper", "baby diaper", "sanitary pad pack", "disposable diaper pile"],
    "paint_chemical": ["paint can", "paint bucket", "pesticide bottle", "chemical bottle label", "spray paint can"],
    "yard_waste": ["dry leaves pile", "grass clippings", "garden waste", "fallen leaves", "tree branches pile"],
}

def count(folder):
    return sum(1 for _ in folder.glob("*") if _.is_file())

ENGINES = (BingImageCrawler, BaiduImageCrawler, GoogleImageCrawler)

for cls, phrases in QUERIES.items():
    out = ROOT / cls
    out.mkdir(parents=True, exist_ok=True)
    for ph in phrases:
        if count(out) >= STOP_AT:
            break
        for Engine in ENGINES:
            try:
                c = Engine(storage={"root_dir": str(out)},
                           downloader_threads=8, log_level=logging.ERROR)
                c.crawl(keyword=ph, max_num=PER_QUERY, min_size=(200, 200),
                        file_idx_offset="auto")
            except Exception as e:
                print("  skipped", Engine.__name__, ph, "->", e, flush=True)
            time.sleep(3)
    print(f"{cls:<18} now {count(out)} files", flush=True)

print("DONE")
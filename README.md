# WasteWise: Waste Segregation System

A mobile app that uses the phone camera to recognise waste items in real time and tells the user which bin to put them in (with disposal guidelines).

**Stack:** Flutter (app) · Python + Flask (API) · HTTP + JSON (communication) · TensorFlow/Keras (MobileNetV2 classifier) · YOLO11 (object detector) · MySQL (planned)

> Status: Phase 1 complete (live detection, classification, bin lookup, disposal guidelines). Phase 2 (MySQL repository, municipal collection schedules) is planned.

---

## How it works

```
Flutter app (camera, 1 frame every ~0.8 s)
        |  HTTP POST /classify (JPEG)
        v
Flask API
  1. Preprocessing        resize 224x224, MobileNetV2 scaling
  2. General model        pretrained ImageNet MobileNetV2: pens and electronics get priority
  3. Detection (YOLO11)   finds up to 3 waste objects, returns bounding boxes
  4. Classification       custom MobileNetV2 refines each cropped box (21 classes)
                          falls back to YOLO's own label if the classifier is unsure
  5. Bin lookup           item -> bin colour + disposal guideline
        |  JSON (item, bin, colour, guideline, confidence, detections[])
        v
Flutter app: coloured bounding boxes + bottom card with bin colour and "How to dispose"
```

If YOLO finds no box, the whole frame goes to the custom classifier, then to the ImageNet fallback.

### Bin colours
| Colour | Meaning |
|---|---|
| Green | Wet / biodegradable |
| Blue | Dry / recyclable |
| Red | Hazardous and e-waste |
| Black | Non-recyclable / general waste |

Colours follow the usual Indian scheme and can be adjusted per municipality.

---

## Project structure

```
waste-segregation/
├── backend/
│   ├── app.py              Flask API (/classify, /health), YOLO + classifier + bin lookup
│   ├── build_dataset.py    merges Kaggle + scraped images into data/processed (224x224, deduplicated)
│   ├── train_model.py      two-phase MobileNetV2 fine-tuning, saves model + labels.json
│   ├── evaluate.py         item-level and bin-level accuracy, per-class results, worst confusions
│   ├── scrape.py           Bing image scraper (icrawler)
│   ├── scrape_daily.py     larger scraper (Bing/Baidu/Google) with per-class phrases
│   ├── extract_frames.py   extracts frames from videos in data/videos into data/scraped
│   ├── preprocess.py       v1 preprocessing (4-class version, kept for reference)
│   └── model/
│       ├── labels.json     21 class names for the classifier
│       ├── waste_classifier.keras   (not in Git, regenerate with train_model.py)
│       └── yolo_waste.pt            (not in Git, trained on Colab)
├── flutter_app/            Flutter live scanner (lib/main.dart)
├── frontend_old/           v1 web upload page (replaced by the app)
├── data/                   datasets (not in Git)
└── README.md
```

---

## Models

### 1. Custom classifier (MobileNetV2, transfer learning)
- 21 classes: background, battery, carton, ceramic, diaper_sanitary, egg_shells_tea, ewaste, food_waste, glass, metal_can, paint_chemical, paper, paper_cup, pen, pencil, plastic_bag, plastic_bottle, plastic_container, plastic_cutlery, thermocol, yard_waste.
- Trained in two phases: classification head only, then fine-tuning the top 30 layers.
- Measured on the validation split: **69% item-level accuracy, 80.5% bin-level accuracy** (`python backend/evaluate.py`).
- The `background` class (empty desks, hands, walls) stops the live camera from labelling everything as an item.

### 2. Detector (YOLO11s)
- Trained on Google Colab (T4 GPU) on a combined object-detection dataset of about 10.8k training images and 12 classes.
- Validation: **mAP50 0.57, mAP50-95 0.46**.
- Strongest classes: stationery (0.75), paper cup (0.70), plastic (0.67), glass (0.64), metal (0.62), plastic bag (0.63).
- Weakest: fabric (0.31), thermocol (0.35), plastic container (0.44). These are partly covered by the classifier.
- Not in the detector's dataset (covered by the classifier and ImageNet layers): battery, e-waste, food waste, ceramic, diaper.

### 3. ImageNet fallback
A pretrained MobileNetV2 (1000 everyday objects). It gives priority to pens and electronics (phones, remotes, keyboards and similar, mapped to the red e-waste bin) and handles some items the other models miss (bottles, fruit).

### Tunable settings (`backend/app.py`)
| Setting | Default | Effect |
|---|---|---|
| `THRESHOLD` | 0.7 | minimum confidence for the custom classifier |
| `GENERAL_THRESHOLD` | 0.20 | minimum for the ImageNet fallback |
| `OVERRIDE_THRESHOLD` | 0.25 | minimum for the pen/electronics priority check |
| `YOLO_CONF` | 0.35 | detector sensitivity: lower = more boxes, higher = fewer wrong ones |

---

## Setup (Mac / Linux)

### Backend
```bash
cd waste-segregation
python3.11 -m venv venv
source venv/bin/activate
pip install flask flask-cors pillow tensorflow ultralytics opencv-python icrawler kaggle
pip install "numpy==1.26.4"      # ultralytics upgrades numpy; TensorFlow 2.16 needs numpy < 2
```
Check everything imports together:
```bash
python -c "import numpy, tensorflow, torch, ultralytics; print('ok')"
```

### Get the datasets
Datasets are not stored in Git. With a Kaggle API token in `~/.kaggle/kaggle.json`:
```bash
mkdir -p data/raw data/scraped && cd data/raw
kaggle datasets download -d phenomsg/waste-classification && unzip -q waste-classification.zip && rm waste-classification.zip
# optional: the 'Garbage Classification' set (mostafaabla) for more images
cd ../..
python backend/scrape.py          # optional extra images into data/scraped
python backend/build_dataset.py   # builds data/processed/train and /val
```

### Train and evaluate the classifier
```bash
python backend/train_model.py     # saves backend/model/waste_classifier.keras + labels.json
python backend/evaluate.py        # item accuracy, bin accuracy, worst confusions
```

### Get the YOLO weights
Train the YOLO11s detector (Colab T4 recommended) and save the result as `backend/model/yolo_waste.pt`.

### Run the API
```bash
python backend/app.py             # wait for "Models ready"; runs on port 5050
curl http://localhost:5050/health
curl -F image=@photo.jpg http://localhost:5050/classify
```
If Flask crashes on Mac when TensorFlow and PyTorch load together, run `KMP_DUPLICATE_LIB_OK=TRUE python backend/app.py`.

### Run the Flutter app
```bash
cd flutter_app
flutter pub get
flutter devices
```
**Android phone over USB (recommended):** keep `apiUrl = 'http://localhost:5050/classify'` in `lib/main.dart` and create a tunnel from the phone to the Mac:
```bash
adb reverse tcp:5050 tcp:5050     # repeat after every unplug/replug
flutter run -d <device-id>
```
If `adb` is not found, use `~/Library/Android/sdk/platform-tools/adb`.

**Phone over Wi-Fi:** set `apiUrl` to `http://<your-mac-ip>:5050/classify` (`ipconfig getifaddr en0`); phone and Mac must be on the same network.

**Chrome:** `flutter run -d chrome` with `localhost` works as is.

Any connection problem is shown in the pill at the top of the app screen.

> **Note for teammates:** `flutter_app/android/gradle/wrapper/gradle-wrapper.properties` points `distributionUrl` at a Gradle mirror, because the default server was blocked on our network. On a normal network you can switch it back to the `services.gradle.org` URL.

---

## API

`POST /classify` with multipart field `image` (JPEG/PNG).

```json
{
  "item": "plastic_bottle",
  "name": "Plastic bottle",
  "bin": "Dry / Recyclable",
  "color": "Blue",
  "guideline": ["Empty it and give it a quick rinse.", "Let it dry and flatten it if possible.", "Put it in the blue bin."],
  "tip": "Put it in the blue bin.",
  "confidence": 0.91,
  "source": "yolo+custom",
  "box": [0.21, 0.30, 0.58, 0.88],
  "detections": [ { "...": "one entry per detected object, box = [x1,y1,x2,y2] as fractions of the image" } ]
}
```
When nothing is recognised, `item` is `"none"` and `detections` is empty. `GET /health` returns the loaded class list.

---

## Known limitations
- Similar items get confused (paper vs plastic bags, glass vs plastic). Best results come from a plain background, good light and one item at a time.
- Frames are processed on the server, so boxes lag by about a second.
- Thermocol, fabric and plastic containers are weak in the detector.
- Scraped training images are noisy; accuracy improves as the dataset grows.

## Roadmap (Phase 2)
- MySQL repository: item-to-bin mappings, disposal guidelines, scan logs.
- Municipal module: collection schedules (daily wet waste, weekly recyclables, monthly hazardous, annual e-waste) and local bin rules.
- On-device inference (TFLite) to remove the server round trip.

## SDGs and POs
SDG 11 (Sustainable Cities and Communities), SDG 12 (Responsible Consumption and Production), SDG 13 (Climate Action). PO6: Engineer and Society. PO7: Environment and Sustainability.

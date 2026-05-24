# Plan implementacji — BoundingBoxer

## Cel

Stworzenie aplikacji do automatycznego anotowania ~2200 zdjęć dłoni.
Dane wejściowe: obrazy posegregowane w folderach `closed_fist/`, `open_palm/`, `none/`.
Potrzebny output: bounding boxy w formacie YOLO (opcjonalnie COCO).

## Harmonogram

| Krok | Moduł | Pliki | Szacowany czas | Status |
|---|---|---|---|---|
| 1 | Konfiguracja i loader | `config.py`, `loader.py`, `detector.py` | 30 min | done |
| 2 | Ekstrakcja i klasyfikacja | `extractor.py`, `classifier.py` | 45 min | done |
| 3 | Eksport i raport | `exporter.py`, `reporter.py` | 30 min | done |
| 4 | CLI | `main.py` | 20 min | done |
| 5 | Streamlit UI | `review/app.py` | 1.5h | done |
| 6 | Testy i kalibracja | — | 30 min | done |

## Architektura — przepływ danych

```
[Obrazy w folderach klas]
         │
         ▼
    ImageLoader           ← config.py (ścieżki, klasy)
         │
         ▼
    HandDetector          ← MediaPipe Hands
         │                   zwraca: landmarki + detection_score
         ▼
    BBoxExtractor         ← z landmarków wylicza bounding box
         │                   format: YOLO (normalizowany 0-1)
         ▼
    GestureClassifier     ← klasyfikacja regułowa z geometrii
         │                   porównuje z oczekiwaną klasą
         ▼
    Reporter              ← generuje report.json
         │                   wylicza combined_confidence
         │                   flaguje do recenzji
         ▼
    Exporter              ← zapisuje .txt (YOLO) lub JSON (COCO)
         │                   + dataset.yaml
         ▼
  ┌─[output/]────────────┐
  │ processed & flagged   │
  └──────────┬────────────┘
             │
             ▼
    Review UI (Streamlit)  ← ładuje report.json + obrazy
         │                   umożliwia korektę
         ▼
    Exporter (ponownie)    ← nadpisuje anotacje po recenzji
```

## Faza 1: Konfiguracja i Loader

### config.py

```python
# Klasy gestów
CLASS_NAMES = ["closed_fist", "open_palm", "none"]

# Powiązanie folder → class_id
CLASS_MAP = {
    "closed_fist": 0,
    "open_palm": 1,
    "none": 2,
}

# Progi
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
COMBINED_CONFIDENCE_THRESHOLD = 0.8   # poniżej → flagowane do recenzji
BBOX_PADDING = 0.10                   # 10% marginesu na bounding box

# Formaty eksportu
SUPPORTED_EXPORT_FORMATS = ["yolo", "coco"]
DEFAULT_EXPORT_FORMAT = "yolo"

# Rozszerzenia obrazów
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Ścieżki wyjściowe
OUTPUT_IMAGES_DIR = "images"
OUTPUT_LABELS_DIR = "labels"
OUTPUT_DATASET_YAML = "dataset.yaml"
OUTPUT_REPORT_JSON = "report.json"
```

### loader.py

```python
class ImageLoader:
    """
    Wczytuje obrazy z katalogu wejściowego.
    Oczekuje struktury: input_dir/class_name/*.jpg
    
    Metody:
    - scan() → list[ImageRecord]         # skanuje wszystkie obrazy
    - load(path) → np.ndarray            # wczytuje obraz (BGR, OpenCV)
    """
```

`ImageRecord` — dataclass:
```python
@dataclass
class ImageRecord:
    path: Path            # pełna ścieżka do pliku
    class_name: str       # nazwa klasy z folderu
    class_id: int         # numeryczne ID klasy
```

### detector.py

```python
class HandDetector:
    """
    Wrapper na mediapipe.solutions.hands.Hands.
    
    Metody:
    - detect(image: np.ndarray) → list[HandDetection]
    """
```

`HandDetection` — dataclass:
```python
@dataclass
class HandDetection:
    landmarks: np.ndarray        # 21x3 (x, y, z) — znormalizowane do [0, 1]
    handedness: str              # "Left" lub "Right"
    detection_score: float       # confidence z MediaPipe
```

## Faza 2: Ekstrakcja i klasyfikacja

### extractor.py

```python
class BBoxExtractor:
    """
    Konwertuje landmarki (21 punktów) na bounding box.
    
    Metody:
    - extract(detection: HandDetection, image_width: int, image_height: int) → BBox
    - to_yolo(bbox: BBox, img_w: int, img_h: int) → tuple[cx, cy, w, h]
    - to_coco(bbox: BBox) → dict      # x, y, width, height (piksele)
    """
```

`BBox` — dataclass:
```python
@dataclass
class BBox:
    x: float      # lewy górny róg (piksele)
    y: float
    width: float
    height: float
    class_id: int
    class_name: str
```

Algorytm ekstrakcji:
1. Znajdź `min_x, min_y, max_x, max_y` ze wszystkich landmarków
2. Oblicz `w = max_x - min_x`, `h = max_y - min_y`
3. Dodaj padding: `w_pad = w * BBOX_PADDING`, `h_pad = h * BBOX_PADDING`
4. Rozszerz box o padding, przycinając do granic obrazu

### classifier.py

```python
class GestureClassifier:
    """
    Klasyfikacja regułowa gestów na podstawie geometrii landmarków.
    
    Metody:
    - classify(landmarks: np.ndarray) → tuple[str, int, float]
      Zwraca: (class_name, class_id, confidence)
    """
```

Algorytm:
1. Dla każdego z 4 palców (bez kciuka): sprawdź czy wyprostowany
   - `extended = distance(MCP, TIP) > distance(MCP, PIP) * 1.3`
2. Sprawdź kciuk osobno (kąt między THUMB_TIP, THUMB_MCP, INDEX_MCP)
3. Zastosuj reguły:
   - `extended_count >= 3` → OPEN_PALM
   - `extended_count == 0` → CLOSED_FIST
   - w przeciwnym razie → UNCERTAIN (confidence = 0.3)

```
Landmarki według MediaPipe:
0:  WRIST
1:  THUMB_CMC
2:  THUMB_MCP
3:  THUMB_IP
4:  THUMB_TIP
5:  INDEX_FINGER_MCP
6:  INDEX_FINGER_PIP
7:  INDEX_FINGER_DIP
8:  INDEX_FINGER_TIP
9:  MIDDLE_FINGER_MCP
10: MIDDLE_FINGER_PIP
11: MIDDLE_FINGER_DIP
12: MIDDLE_FINGER_TIP
13: RING_FINGER_MCP
14: RING_FINGER_PIP
15: RING_FINGER_DIP
16: RING_FINGER_TIP
17: PINKY_MCP
18: PINKY_PIP
19: PINKY_DIP
20: PINKY_TIP
```

## Faza 3: Eksport i raport

### exporter.py

```python
class Exporter:
    """
    Zapisuje anotacje w formacie YOLO lub COCO.
    
    Metody:
    - export_yolo(results: list[ProcessingResult], output_dir: Path)
    - export_coco(results: list[ProcessingResult], output_dir: Path)
    - generate_dataset_yaml(output_dir: Path, class_names: list[str])
    """
```

`ProcessingResult` — łączy `ImageRecord` + wyniki detekcji:
```python
@dataclass
class ProcessingResult:
    image_record: ImageRecord
    detections: list[HandDetection]
    bboxes: list[BBox]
    detected_class: str | None
    detected_class_id: int | None
    classification_confidence: float
    mediapipe_confidence: float
    combined_confidence: float
    needs_review: bool
    reviewed: bool = False
    manual_override: bool = False
```

### reporter.py

```python
class Reporter:
    """
    Generuje report.json z wynikami przetwarzania.
    
    Metody:
    - generate(results: list[ProcessingResult]) → dict
    - save(report: dict, output_path: Path)
    - load(report_path: Path) → dict
    - get_summary(report: dict) → Summary  # statystyki
    """
```

`Summary`:
```python
@dataclass
class Summary:
    total_images: int
    total_detected: int
    total_not_detected: int
    total_reviewed: int
    total_needs_review: int
    average_confidence: float
    per_class_stats: dict[str, ClassStats]
```

## Faza 4: CLI

### main.py

Dwa główne polecenia:

```bash
python -m boundingboxer process [opcje]
python -m boundingboxer review [opcje]
```

Struktura komend (argparse):

```
boundingboxer
├── process
│   --input          (wymagany)  ścieżka do katalogu z danymi
│   --output         (wymagany)  ścieżka wyjściowa
│   --format         (opcjonalny) "yolo" | "coco"  [default: yolo]
│   --confidence     (opcjonalny) próg confidence  [default: 0.8]
│
└── review
    --input          (wymagany)  ścieżka do katalogu output z report.json
    --port           (opcjonalny) port Streamlit     [default: 8501]
```

## Faza 5: Streamlit UI

### review/app.py

Struktura interfejsu:

```
┌─────────────────────────────────────────────┐
│  🔍 BoundingBoxer Review                    │
│  ─────────────────────────────────────────  │
│  [Sidebar]                │  [Main Panel]   │
│                           │                 │
│  Filtry:                  │  Obraz 42/320   │
│  ┌─────────────────┐     │  ┌───────────┐  │
│  │ Confidence: 0.7 │     │  │           │  │
│  │ [====●======]   │     │  │  Obraz z  │  │
│  └─────────────────┘     │  │  bboxem   │  │
│                           │  │           │  │
│  ☑ Tylko do przejrzenia  │  └───────────┘  │
│  ☑ Tylko nieprzejrzane   │                 │
│                           │  Klasa: closed_fist │
│  Klasa: [wszystkie  ▼]   │  Confidence: 0.95   │
│                           │                 │
│  ─────────────────────    │  Edycja boxa:   │
│  Postęp: 192/320 (60%)    │  x: [===●====]  │
│  [████████░░░░░░]         │  y: [===●====]  │
│                           │  w: [===●====]  │
│  ─────────────────────    │  h: [===●====]  │
│                           │                 │
│  [Export reviewed]        │  [◀ Prev] [Approve ▶] │
│                           │                 │
└─────────────────────────────────────────────┘
```

#### Stan sesji (Streamlit session_state)

```python
# Indeks aktualnie wyświetlanego obrazu
current_idx: int = 0

# Lista przefiltrowanych rezultatów (załadowana z report.json)
filtered_results: list[dict]

# Czy wprowadzono zmiany w boksie
bbox_modified: bool = False
```

#### Obsługa korekty

Ponieważ Streamlit nie wspiera natywnie drag&drop na obrazach, edycja odbywa się przez:
1. Suwaki numeryczne (x, y, width, height) w pikselach
2. Podgląd na żywo — obraz z nałożonym bounding boxem aktualizuje się przy każdej zmianie suwaka
3. Przycisk "Reset bbox" przywraca oryginalne wartości z detekcji
4. Po zatwierdzeniu: zapis do report.json z flagą `reviewed=true`

#### Nawigacja

- Przyciski "Prev" / "Next" do przechodzenia między obrazami
- Skrót klawiszowy: Enter → "Approve & Next"
- Dropdown do skoku do konkretnego obrazu

## Faza 6: Testy i kalibracja

### Testy jednostkowe

- `loader.py` — poprawne skanowanie struktury katalogów, rozpoznawanie klas
- `extractor.py` — bounding box z landmarków (stałe dane testowe)
- `classifier.py` — klasyfikacja dla znanych układów dłoni
- `exporter.py` — poprawność formatu YOLO/COCO

### Kalibracja na rzeczywistych danych

1. Uruchomić pipeline na małej próbce (10-20 zdjęć na klasę)
2. Sprawdzić rozkład confidence scores
3. Dostosować próg `COMBINED_CONFIDENCE_THRESHOLD`
4. Dostroić parametry klasyfikacji regułowej (`EXTENDED_RATIO = 1.3`)

### Możliwe problemy

| Problem | Rozwiązanie |
|---|---|
| MediaPipe nie wykrywa dłoni w trudnych warunkach | Obniżenie `MIN_DETECTION_CONFIDENCE`, rotacja/flip obrazu jako augmentacja |
| Bounding box za ciasny / za luźny | Dostrojenie `BBOX_PADDING` |
| Klasyfikacja regułowa myli gesty | Dodanie bardziej zaawansowanych reguł (kąty, wzajemne odległości palców) |
| Wydajność na 2200 obrazach | Przetwarzanie wsadowe z progress barem (tqdm) |

## Kolejne kroki po wdrożeniu

1. Przetworzenie całego datasetu (~2200 zdjęć)
2. Przejrzenie flagowanych obrazów w UI
3. Wyeksportowanie finalnych anotacji YOLO
4. Podział na train/val (np. 80/20)
5. Trenowanie małego modelu YOLO (np. YOLOv8n)

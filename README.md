# BoundingBoxer

Aplikacja do automatycznego anotowania datasetu dłoni z użyciem Google MediaPipe.
Wykrywa dłonie na zdjęciach, generuje bounding boxy w formacie YOLO (lub COCO)
oraz umożliwia manualną recenzję wyników o niskim confidence.

## Problem

Dataset zawiera ~2200 zdjęć dłoni posegregowanych w folderach według klasy gestu
(`closed_fist/`, `open_palm/`, `none/`). Brakuje jednak bounding boxów potrzebnych
do trenowania małych modeli YOLO. BoundingBoxer rozwiązuje ten problem:

1. Wykrywa dłonie za pomocą MediaPipe Hands (21 landmarków)
2. Generuje bounding boxy z landmarków
3. Klasyfikuje gesty na podstawie geometrii dłoni
4. Umożliwia recenzję/korektę wyników o niskim confidence

## Tech Stack

| Technologia | Zastosowanie |
|---|---|
| Python 3.10+ | Główny język |
| MediaPipe | Wykrywanie dłoni i landmarków |
| OpenCV | Wczytywanie obrazów, rysowanie overlayów |
| Streamlit | Web UI do recenzji/korekty manualnej |
| NumPy | Obliczenia numeryczne |
| Pillow | Manipulacja obrazami |
| PyYAML | Generowanie dataset.yaml dla YOLO |

## Instalacja

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Użycie — CLI

### Przetwarzanie wsadowe

```bash
python -m boundingboxer process \
  --input ./data \
  --output ./output \
  --format yolo \
  --confidence-threshold 0.7
```

Argumenty:
- `--input` — ścieżka do katalogu z podfolderami klas (np. `closed_fist/`, `open_palm/`, `none/`)
- `--output` — ścieżka do katalogu wyjściowego
- `--format` — `yolo` (domyślnie) lub `coco`
- `--confidence-threshold` — próg confidence (domyślnie `0.7`), poniżej którego obraz jest flagowany do recenzji

### Uruchomienie UI recenzji

```bash
python -m boundingboxer review \
  --input ./output \
  --port 8501
```

Otwórz przeglądarkę na `http://localhost:8501`.

## Struktura katalogu wyjściowego

```
output/
├── images/              # kopie oryginałów
│   ├── closed_fist/
│   ├── open_palm/
│   └── none/
├── labels/              # pliki YOLO .txt (jeden na obraz)
│   ├── closed_fist/
│   ├── open_palm/
│   └── none/
├── dataset.yaml         # konfiguracja dla YOLOv5/v8
└── report.json          # status przetwarzania każdego obrazu
```

### dataset.yaml

```yaml
path: ./output
train: images
val: images
names:
  0: closed_fist
  1: open_palm
  2: none
```

### report.json (struktura wpisu)

```json
{
  "image": "closed_fist/img001.jpg",
  "detected": true,
  "detected_class": "closed_fist",
  "expected_class": "closed_fist",
  "mediapipe_confidence": 0.95,
  "combined_confidence": 0.95,
  "bbox": [0.45, 0.50, 0.30, 0.40],
  "reviewed": false,
  "needs_review": false
}
```

## Klasyfikacja gestów — metoda regułowa

Klasyfikacja opiera się na geometrii landmarków dłoni (21 punktów MediaPipe):

```
Dla każdego palca (index, middle, ring, pinky):
  extended = distance(MCP, TIP) > distance(MCP, PIP) * 1.3

Jeśli extended_fingers >= 3 → OPEN_PALM
Jeśli extended_fingers == 0 → CLOSED_FIST
W przeciwnym razie → UNCERTAIN (niskie confidence)
```

Dodatkowo sprawdzany jest kciuk (odległość THUMB_TIP od INDEX_MCP).

## Logika confidence

- **`mediapipe_score`** (0–1) — confidence z MediaPipe Hands
- **`gesture_match`** (0 lub 1) — czy sklasyfikowany gest zgadza się z etykietą folderu
- **`combined_confidence`** = `mediapipe_score * gesture_match`

### Przypadki flagowane do recenzji

- `combined_confidence < threshold` (domyślnie 0.8)
- Brak wykrytej dłoni w folderze `closed_fist/` lub `open_palm/`
- Wykryto dłoń w folderze `none/`

## Streamlit UI — recenzja manualna

Interfejs webowy umożliwiający przeglądanie i korektę anotacji.

### Funkcje

| Element UI | Opis |
|---|---|
| Filtr confidence | Slider: pokaż tylko obrazy z confidence < X |
| Filtr statusu | Checkboxy: "tylko do przejrzenia", "tylko nieprzejrzane" |
| Filtr klasy | Dropdown z mismatch-ami (detected ≠ expected) |
| Pasek postępu | "Przejrzano 142/320 flagged" |
| Widok obrazu | Obraz + nałożony bounding box (czerwony = niezatwierdzony, zielony = zatwierdzony) |
| Edycja boxa | Suwaki numeryczne x, y, w, h z live preview |
| Edycja klasy | Dropdown: closed_fist / open_palm / none |
| Dodaj/usuń box | Przyciski |
| Zatwierdź | "Approve & Next" + skrót klawiszowy (Enter) |
| Eksport | Przycisk "Export reviewed annotations" |

## Struktura kodu źródłowego

```
BoundingBoxer/
├── boundingboxer/
│   ├── __init__.py
│   ├── main.py              # CLI entry point (argparse)
│   ├── config.py            # konfiguracja (progi, klasy gestów, formaty)
│   ├── loader.py            # wczytywanie obrazów z podfolderów klas
│   ├── detector.py          # MediaPipe Hands wrapper
│   ├── extractor.py         # landmarki → bounding box (YOLO/COCO)
│   ├── classifier.py        # klasyfikacja regułowa gestów
│   ├── exporter.py          # zapis anotacji do plików
│   ├── reporter.py          # generowanie report.json ze statystykami
│   └── review/
│       ├── __init__.py
│       └── app.py           # Streamlit UI
├── requirements.txt
└── README.md
```

## Zależności (requirements.txt)

```
mediapipe>=0.10.0
opencv-python>=4.8.0
streamlit>=1.28.0
numpy>=1.24.0
Pillow>=10.0.0
pyyaml>=6.0
```

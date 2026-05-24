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
| tqdm | Pasek postępu przetwarzania |

## Instalacja

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Oczekiwana struktura danych wejściowych

```
data/
├── closed_fist/
│   ├── zdjecie1.jpg
│   ├── zdjecie2.jpg
│   └── ...
├── open_palm/
│   ├── zdjecie3.png
│   └── ...
└── none/
    ├── zdjecie4.jpeg
    └── ...
```

- Nazwy folderów **muszą** odpowiadać klasom: `closed_fist`, `open_palm`, `none`
- Obsługiwane rozszerzenia: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- Foldery niepasujące do żadnej klasy są ignorowane
- Pliki o innych rozszerzeniach (`.txt`, `.json`) są pomijane

## Użycie — CLI

### Przetwarzanie wsadowe

```bash
python -m boundingboxer process \
  --input ./data \
  --output ./output \
  --format yolo \
  --confidence 0.8
```

Argumenty:
| Flaga | Wymagany | Domyślnie | Opis |
|-------|----------|-----------|------|
| `--input` | tak | — | Ścieżka do katalogu z podfolderami klas |
| `--output` | tak | — | Ścieżka do katalogu wyjściowego |
| `--format` | nie | `yolo` | Format eksportu: `yolo` lub `coco` |
| `--confidence` | nie | `0.8` | Próg combined confidence — poniżej obraz jest flagowany do recenzji |

Podczas przetwarzania wyświetlany jest pasek postępu (`tqdm`). Błędy dla pojedynczych obrazów są logowane do stderr, ale nie przerywają całego pipeline'a.

### Uruchomienie UI recenzji

```bash
python -m boundingboxer review \
  --input ./output \
  --port 8501
```

Otwórz przeglądarkę na `http://localhost:8501`.

## Pełny przepływ pracy

### Krok 1: Przetwarzanie automatyczne

```bash
python -m boundingboxer process --input ./data --output ./output
```

Pipeline:
1. Skanuje foldery `data/closed_fist/`, `data/open_palm/`, `data/none/`
2. Dla każdego obrazu: wykrywa dłonie (MediaPipe), generuje bounding boxy, klasyfikuje gest
3. Oblicza `combined_confidence` i flaguje obrazy wymagające recenzji
4. Zapisuje wyniki do `output/`:
   - `report.json` — status każdego obrazu
   - `labels/*/` — pliki `.txt` w formacie YOLO
   - `dataset.yaml` — konfiguracja dla YOLOv5/v8

### Krok 2: Recenzja manualna

```bash
python -m boundingboxer review --input ./output
# Otwórz http://localhost:8501
```

W interfejsie:
1. Użyj filtrów w sidebarze, żeby pokazać tylko obrazy wymagające uwagi
2. Przeglądaj obrazy strzałkami "◀ Previous" / "Approve & Next"
3. W razie potrzeby popraw bounding box (suwaki X, Y, Width, Height) lub klasę
4. Kliknij **"Approve & Next"** — zapisuje zmiany, oznacza `reviewed = true` i przechodzi dalej
5. Powtarzaj aż wszystkie flagowane obrazy zostaną przejrzane

### Krok 3: Eksport i trenowanie

Po recenzji masz kompletny dataset YOLO w `output/`. Możesz od razu trenować:

```bash
yolo detect train data=output/dataset.yaml model=yolov8n.pt epochs=100
```

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
  "classification_confidence": 0.90,
  "combined_confidence": 0.95,
  "bbox": [0.45, 0.50, 0.30, 0.40],
  "reviewed": false,
  "needs_review": false,
  "manual_override": false
}
```

Pola:
| Pole | Typ | Opis |
|------|-----|------|
| `image` | string | Ścieżka względna do obrazu (np. `"closed_fist/img001.jpg"`) |
| `detected` | bool | Czy MediaPipe wykrył dłoń |
| `detected_class` | string/null | Klasa sklasyfikowana przez `GestureClassifier` (lub null) |
| `expected_class` | string | Klasa oczekiwana (nazwa folderu, z którego pochodzi obraz) |
| `mediapipe_confidence` | float | Pewność detekcji z MediaPipe Hands |
| `classification_confidence` | float | Pewność klasyfikacji regułowej (0.9 lub 0.3) |
| `combined_confidence` | float | `mediapipe_confidence * (1.0 jeśli zgodna klasa, 0.0 jeśli nie)` |
| `bbox` | list/null | Bounding box w formacie YOLO `[cx, cy, w, h]` lub null |
| `reviewed` | bool | Czy użytkownik zatwierdził anotację w UI |
| `needs_review` | bool | Czy obraz wymaga recenzji (niski confidence lub brak detekcji) |
| `manual_override` | bool | Czy użytkownik ręcznie zmienił bbox lub klasę |

## Klasyfikacja gestów — metoda regułowa

Klasyfikacja opiera się na geometrii landmarków dłoni (21 punktów MediaPipe):

```
Dla każdego palca (index, middle, ring, pinky):
  extended = distance(MCP, TIP) > distance(MCP, PIP) * 1.3

Jeśli extended_fingers >= 3 → OPEN_PALM (confidence 0.9)
Jeśli extended_fingers == 0 → CLOSED_FIST (confidence 0.9)
W przeciwnym razie → "none" (confidence 0.3)
```

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
| Filtr confidence | Slider: pokaż tylko obrazy z combined_confidence ≥ X |
| Filtr "Only needs review" | Checkbox: pokaż tylko obrazy wymagające recenzji |
| Filtr "Only unreviewed" | Checkbox: pokaż tylko obrazy jeszcze nieprzejrzane |
| Filtr klasy | Dropdown: "all" / "closed_fist" / "open_palm" / "none" |
| Pasek postępu | "Reviewed 142/320 (44%)" — pokazuje postęp recenzji wśród przefiltrowanych |
| Widok obrazu | Obraz + nałożony bounding box (czerwony = niezatwierdzony, zielony = zatwierdzony) |
| Edycja boxa | 4 pola numeryczne: X, Y, Width, Height (w pikselach) |
| Edycja klasy | Dropdown: closed_fist / open_palm / none |
| Reset bbox | Przycisk przywracający oryginalny bbox z detekcji |
| **Approve & Next** | Zapisuje zmiany (bbox + klasa), oznacza jako `reviewed`, przechodzi do następnego |

### Jak korzystać z filtrów

Najlepsza strategia:
1. Zaznacz ☑ "Only needs review" + ☑ "Only unreviewed" — pokazuje tylko obrazy, które automatycznie zostały zakwalifikowane do recenzji
2. Przejrzyj je wszystkie, poprawiając bboxy tam gdzie trzeba
3. Odznacz "Only needs review", ustaw suwak confidence na np. 0.6 — sprawdź czy reszta wygląda OK
4. Na końcu odznacz wszystko i przejrzyj cały dataset dla pewności

### Uwagi

- Każde kliknięcie "Approve & Next" natychmiast zapisuje zmiany do `report.json`
- Pole `manual_override` jest automatycznie ustawiane na `true` jeśli zmieniłeś bbox lub klasę
- "Export Reviewed" (sidebar) to placeholder — po recenzji dane YOLO są już wyeksportowane, wystarczy odpalić `process` ponownie z tym samym `--output` żeby uwzględnić tylko przejrzane obrazy
- Streamlit nie wspiera drag&drop na obrazach — edycja odbywa się suwakami numerycznymi

## Struktura kodu źródłowego

```
BoundingBoxer/
├── boundingboxer/
│   ├── __init__.py
│   ├── __main__.py           # python -m boundingboxer
│   ├── main.py               # CLI entry point, pipeline orchestrator
│   ├── config.py             # konfiguracja (progi, klasy, formaty)
│   ├── loader.py             # wczytywanie obrazów z podfolderów klas
│   ├── detector.py           # MediaPipe Hands wrapper
│   ├── extractor.py          # landmarki → bounding box (YOLO/COCO)
│   ├── classifier.py         # klasyfikacja regułowa gestów
│   ├── exporter.py           # zapis anotacji do plików YOLO/COCO
│   ├── reporter.py           # generowanie report.json ze statystykami
│   └── review/
│       ├── __init__.py
│       ├── logic.py          # czyste funkcje (filtrowanie, konwersja bbox)
│       └── app.py            # Streamlit UI
├── tests/
│   ├── test_loader.py
│   ├── test_detector.py
│   ├── test_extractor.py
│   ├── test_classifier.py
│   ├── test_exporter.py
│   ├── test_reporter.py
│   ├── test_main.py
│   ├── test_review.py
│   └── test_integration.py   # testy end-to-end
├── requirements.txt
└── README.md
```

## Troubleshooting

### MediaPipe nie wykrywa dłoni na wielu obrazach

Obniż próg detekcji w `boundingboxer/config.py`:
```python
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3   # domyślnie 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.3    # domyślnie 0.5
```

### Bounding box za ciasny / za luźny

Dostrój margines w `config.py`:
```python
BBOX_PADDING = 0.15   # domyślnie 0.10 (10%), zwiększ jeśli box za ciasny
```

### Klasyfikacja regułowa myli gesty (open_palm vs closed_fist)

Dostrój próg wyprostowania palca w `boundingboxer/classifier.py`:
```python
EXTENDED_RATIO = 1.1   # domyślnie 1.3, zmniejsz jeśli palce uznawane za zgięte
                       # zwiększ jeśli zgięte uznawane za wyprostowane
```

### Za dużo obrazów flagowanych do recenzji

Podnieś próg w `config.py` lub podaj niższy przez CLI:
```python
COMBINED_CONFIDENCE_THRESHOLD = 0.6   # domyślnie 0.8
```
```bash
python -m boundingboxer process --input ./data --output ./output --confidence 0.6
```

### "No module named 'mediapipe'" przy uruchamianiu

```bash
pip install -r requirements.txt
```

### ImportError: cannot import name 'solutions' from 'mediapipe'

Twoja wersja MediaPipe (≥0.10) nie ma legacy API `mp.solutions`. Zainstaluj starszą:
```bash
pip install "mediapipe<0.10"
```
Lub użyj obecnej wersji — testy omijają ten problem przez mockowanie, a pipeline używa `mp.solutions.hands.Hands`.

## Zależności

```
mediapipe>=0.10.0
opencv-python>=4.8.0
streamlit>=1.28.0
numpy>=1.24.0
Pillow>=10.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

Pełna lista w `requirements.txt`.

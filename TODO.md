# BoundingBoxer — Plan rozbudowy Streamlit GUI

## Cel
Rozbudowa istniejącego `review/app.py` o tryb **Process** — pełny setup, odpalenie pipeline'a, pasek postępu na żywo i podsumowanie. Po zakończeniu jednym kliknięciem przejście do trybu Review.

---

## Screen layout

```
┌─ Sidebar ───────────────────────┬─ Main ──────────────────────────┐
│                                 │                                 │
│  Mode: [Process ▼]              │  ⚙️ Pipeline                     │
│                                 │                                 │
│  📁 Input:  [./data]  [Browse] │  ┌─────────────────────────┐   │
│  📁 Output: [./output][Browse] │  │ Processing...            │   │
│                                 │  │ ████████████░░░░ 67%    │   │
│  Format:   [yolo ▼]            │  │ 6/9 images              │   │
│  Confid.:  [======●===] 0.8   │  └─────────────────────────┘   │
│                                 │                                 │
│  [▶ Run Pipeline]               │  ✅ Done!                       │
│                                 │  ┌──────┬──────┬──────────┐   │
│  ──────────────────────         │  │Class │Total │Detected  │   │
│                                 │  │cf    │  3   │   1      │   │
│  Przełącz na:                   │  │op    │  3   │   2      │   │
│  [Open in Review]               │  │none  │  3   │   0      │   │
│                                 │  └──────┴──────┴──────────┘   │
│                                 │  [Open in Review →]             │
└─────────────────────────────────┴─────────────────────────────────┘
```

## Zakres zmian

| Plik | Zmiana | Szacunek |
|------|--------|----------|
| `main.py` | Dodać `progress_callback` do `run_pipeline()` | 5 min |
| `review/app.py` | Dodać `st.session_state.mode`, panel Process, spinner/progress, tabelę podsumowania | 45 min |
| `review/logic.py` | Nowa funkcja `build_summary_table(report)` | 5 min |
| `tests/test_review.py` | Testy dla `build_summary_table` | 10 min |
| `tests/test_main.py` | Test dla `progress_callback` | 10 min |

---

## Flow użytkownika

1. Odpala `python -m boundingboxer review` (bez argumentów)
2. Domyślnie tryb **Process** — wypełnia input/output, klika **Run Pipeline**
3. Widzi pasek postępu aktualizowany na żywo
4. Po zakończeniu — tabela podsumowania (per klasa: total, detected, not detected, avg confidence)
5. Klika **Open in Review** → przechodzi do trybu Review z załadowanym report.json
6. Recenzuje, zatwierdza, klika Approve & Next
7. Gotowe

---

## Technikalia

### `main.py` — progress_callback
```python
def run_pipeline(input_dir, output_dir, format="yolo", confidence_threshold=0.8,
                 progress_callback=None):
    ...
    for i, record in enumerate(tqdm(records, desc="Processing")):
        ...
        if progress_callback:
            progress_callback(i + 1, len(records))
```
W CLI tqdm działa jak dotychczas. W Streamlit callback aktualizuje `st.progress()`.

### `review/app.py` — session_state
```python
if "mode" not in st.session_state:
    st.session_state.mode = "process"
if "process_report" not in st.session_state:
    st.session_state.process_report = None
```

### `review/app.py` — Process mode sidebar
```python
if st.session_state.mode == "process":
    st.sidebar.selectbox("Mode", ["Process", "Review"], key="mode")
    input_dir = st.sidebar.text_input("Input", value="./data")
    output_dir = st.sidebar.text_input("Output", value="./output")
    export_format = st.sidebar.selectbox("Format", ["yolo", "coco"])
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8)
    if st.sidebar.button("▶ Run Pipeline"):
        # run pipeline with progress updates
        ...
```

### `review/app.py` — Progress display
```python
progress_bar = st.progress(0)
status_text = st.empty()

def progress_callback(current, total):
    progress_bar.progress(current / total)
    status_text.text(f"Processing {current}/{total} images...")

report = run_pipeline(input_dir, output_dir, format=export_format,
                       confidence_threshold=confidence,
                       progress_callback=progress_callback)
```

### `review/app.py` — Summary table
```python
st.success("✅ Processing complete!")
from ..reporter import Reporter
summary = Reporter().get_summary(report)
# Display summary table using st.dataframe() or st.table()
```

### `review/app.py` — Switch to Review
```python
if st.button("Open in Review →"):
    st.session_state.mode = "review"
    st.session_state.review_input = output_dir
    st.rerun()
```

### `review/logic.py` — build_summary_table
```python
def build_summary_table(report):
    """Zwraca listę krotek (class_name, total, detected, not_detected, avg_conf)."""
    ...

```

---

## Kolejność implementacji

| Krok | Plik | Agent | Status |
|------|------|-------|--------|
| 1 | `main.py` — dodać `progress_callback` | @coder | done |
| 2 | `review/logic.py` — `build_summary_table()` | @coder | done |
| 3 | Testy dla `build_summary_table` + `progress_callback` | @tester | done |
| 4 | `review/app.py` — tryb Process + pasek postępu + podsumowanie | @coder | done |
| 5 | Review całości | @karpathy_reviewer | done |
| 6 | Final test run (257 → 265) | pytest | done |
| 7 | Poprawki: `--input` optional dla review, importy absolute zamiast relative | @coder | done |
| 8 | Fix widget state `mode` → `app_mode` + `mode_widget` | @coder | done |
| 9 | `exporter.export_images()` kopiuje obrazy do output/ | @coder | done |

---

# TODO 2: Poprawki UX + rysowanie bbox myszką + kalibracja detekcji

## Cel

Naprawić bug z "Reset bbox", dodać rysowanie bounding boxa myszką na podglądzie, dodać regulację `detection_confidence` w GUI.

---

## A. Fix "Reset bbox" (widget state conflict)

**Bug:** `_reset_entry_state` próbuje pisać do `st.session_state.bbox_x/y/w/h`, ale te klucze należą do widgetów
`st.number_input(key="bbox_x")` → `StreamlitAPIException`.

**Rozwiązanie: separacja kluczy widgetów od app state (ten sam wzorzec co `app_mode`).**

| Cel | Klucz widgetu | Klucz app state |
|-----|--------------|-----------------|
| X | `bbox_x_widget` | `app_bbox_x` |
| Y | `bbox_y_widget` | `app_bbox_y` |
| W | `bbox_w_widget` | `app_bbox_w` |
| H | `bbox_h_widget` | `app_bbox_h` |
| Klasa | `class_override_widget` | `app_class_override` |

Widgety używają `value=` do inicjalizacji z app state. `on_change` synchronizuje widget→app. Kod aplikacji czyta/pisze tylko app state.

**Zmiany w `app.py`:** `_on_bbox_change`, `_reset_entry_state`, widgety number_input, selectbox, "Approve & Next" — ~25 linijek.

---

## B. Mouse-drawn bounding box na podglądzie

**Technologia: `streamlit-drawable-canvas`**

Biblioteka oparta na Fabric.js (HTML5 Canvas). Wspiera rysowanie prostokątów, zwraca współrzędne jako JSON.

**Flow:**
1. Obraz wyświetlany na `<canvas>` z opcją `drawing_mode="rect"`
2. Użytkownik rysuje prostokąt myszką
3. `canvas.json_data` → `{left, top, width, height}` w pikselach
4. Konwersja na YOLO `[cx, cy, w, h]` (normalized 0-1)
5. Zapis do `app_bbox_x/y/w/h` → synchronizuje suwaki numeryczne
6. Suwaki numeryczne dalej dostępne jako alternatywna edycja

**Zmiany:**
| Plik | Zmiana |
|------|--------|
| `requirements.txt` | Dodać `streamlit-drawable-canvas` |
| `review/app.py` | Zastąpić `st.image()` canvasem, dodać `canvas_data → bbox` konwersję |

---

## C. Detection confidence slider w Process mode

Dodać drugi slider w sidebarze Process mode — `detection_confidence` (domyślnie 0.5, zakres 0.1–1.0), przekazywany do `HandDetector` przez `run_pipeline()`.

**Zmiany:**
| Plik | Zmiana |
|------|--------|
| `main.py` | `run_pipeline()` przyjmuje `detection_confidence` i przekazuje do `HandDetector()` |
| `review/app.py` | Sidebar: slider `detection_confidence` obok `confidence_threshold` |

---

## D. Usprawnienia detekcji (przyszłość)

Po przetworzeniu datasetu:
- Sprawdzić rozkład `detected=False` per klasa
- Dla obrazów bez detekcji: obniżyć próg do 0.3, powtórzyć pipeline
- Rozważyć dodanie flip-horizontal augmentacji (detekcja na oryginale + flipie)
- Po ręcznej review ~200 obrazów: trenować Random Forest na landmarkach zamiast reguł

---

## Kolejność implementacji

| Krok | Plik | Zadanie | Status |
|------|------|---------|--------|
| 1 | `review/app.py` | Fix "Reset bbox" — separacja kluczy widgetów (wzorzec `app_mode`) | done |
| 2 | `review/app.py` + `main.py` | Dodać `detection_confidence` slider w Process mode | done |
| 3 | `review/app.py` + `requirements.txt` | Dodać `streamlit-drawable-canvas` — rysowanie bbox myszką | pending |
| 4 | `tests/test_review.py` | Testy dla nowych funkcji | pending |
| 5 | — | Review + final test run | pending |

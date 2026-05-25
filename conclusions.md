# BoundingBoxer — Post-mortem

---

## 1. Główny błąd: błędne założenie o klasyfikacji regułowej

Założyliśmy, że 4 linijki kodu (`d_mcp_tip > d_mcp_pip * 1.3`) wystarczą do rozróżnienia gestów. Rzeczywistość:

| Problem | Dlaczego zawiodło |
|---------|-------------------|
| Tylko 12 z 63 współrzędnych | Kciuk, nadgarstek, orientacja dłoni — ignorowane całkowicie |
| Odległości 2D bez Z-depth | Dłoń pod kątem → wszystkie odległości skompresowane → reguła nie działa |
| Sztywny próg 1.3 | Inna anatomia, inna odległość od kamery → inny próg |
| "none" vs "uncertain" | Klasa "none" oznaczała zarówno "brak ręki" jak i "nie wiem" |

CLIP miał to naprawić, ale:

| Dlaczego CLIP zawiódł |
|-----------------------|
| Trenowany na zdjęciach internetowych, nie na gestach dłoni — nie rozróżnia subtelnych różnic |
| Crop 50×50 pikseli z 224×224 obrazu → za mało informacji dla ViT-B-32 |
| Kontekst (tło, oświetlenie, kolor skóry) waży więcej niż sam gest |
| Zero-shot ≠ magic. Ten konkretny problem wymaga fine-tuningu |

---

## 2. MediaPipe jako wąskie gardło

| Co | Konsekwencja |
|----|-------------|
| Wykrywa tylko "czyste" dłonie | ~40% obrazów z trudnym oświetleniem/kątem — brak detekcji |
| Brak możliwości fine-tuningu | Nie da się go douczyć na własnych danych |
| `detect_with_flip()` pomogło | Ale tylko częściowo — flip nie naprawia złego kąta ręki |

---

## 3. Streamlit jako zły wybór UI

| Problem | Stracony czas |
|---------|--------------|
| Konflikt widget state vs app state | 5 iteracji debugowania tego samego błędu |
| Brak natywnego rysowania na obrazie | Trzeba było pisać własny komponent canvas (HTML/JS) |
| Live progress bar w pipeline | Wymagał `progress_callback` + `st.rerun()` — kruche |
| Tryb Process + Review w jednej aplikacji | Mieszanie odpowiedzialności, trudne w utrzymaniu |

**Wniosek:** Streamlit jest świetny do dashboardów, nie do interaktywnych narzędzi anotacyjnych.

---

## 4. Co zrobiliśmy dobrze

| Element | Ocena |
|---------|-------|
| Architektura modułowa (detector, classifier, extractor, exporter, reporter) | Dobra |
| 278 testów jednostkowych + integracyjnych | Bardzo dobra |
| Planowanie przez TODO.md z harmonogramem | Dobra |
| Koncepcja aktywnej pętli uczenia (review → retrain) | Dobra — ale nie zaimplementowana |
| Review UI z confidence sortowaniem ASC | Dobra — właściwy UX |
| Eksport YOLO/COCO | Działa poprawnie |

---

## 5. Kluczowe wnioski dla nowej aplikacji

| # | Wniosek |
|---|---------|
| 1 | **Zacznij od ręcznego labelowania 100-200 obrazów.** Daje to dane treningowe i zrozumienie domeny |
| 2 | **Użyj YOLOv8 do detekcji + klasyfikacji w jednym modelu.** Jeden pass, jeden model, brak MediaPipe |
| 3 | **Desktop app, nie Streamlit.** PyQt / Electron / Flask+HTML — cokolwiek z natywnym canvasem i normalnym stanem widgetów |
| 4 | **Aktywna pętla uczenia od pierwszego dnia.** Review 50 → retrain → classify → review mniej |
| 5 | **Oddziel przetwarzanie od recenzji.** CLI tool do batch processingu, osobna apka do review |
| 6 | **Feature engineering z landmarków → ML classifier**, nie reguły. RandomForest na 63+20 cechach geometrii dłoni bije zarówno reguły jak i zero-shot CLIP dla tego zadania |
| 7 | **Zaakceptuj ~5-10% manual review.** Cel to minimalizować, nie eliminować |

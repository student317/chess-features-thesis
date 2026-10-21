# chess-features-thesis

Szkielet pod część (b): klasyfikator cech pozycji + korelacja z ratingiem gracza.

## Struktura

```
chess-features-thesis/
├── requirements.txt
├── data/
│   └── sample.pgn          # 3 syntetyczne partie do testów (castling, en passant, promocja)
├── src/
│   └── features.py         # cała logika ekstrakcji cech - tu dokładacie nowe cechy
└── notebooks/
    └── 01_explore_features.ipynb   # uruchamia pipeline, analiza/korelacje
```

## Setup (lokalnie)

```bash
cd chess-features-thesis
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Otwórz `notebooks/01_explore_features.ipynb` i uruchom komórki po kolei.
Pierwsze uruchomienie powinno przejść na `data/sample.pgn` i pokazać, że
castling, en passant i promocja są poprawnie wykrywane (sanity-checki są
w notebooku).

## Skąd wziąć prawdziwe dane

`data/sample.pgn` to tylko 3 partie do testów pipeline'u. Do właściwej
analizy pobierzcie miesięczny zrzut z https://database.lichess.org/
(pliki `.pgn.zst`, zawierają ratingi obu graczy w nagłówkach `WhiteElo`/`BlackElo`).
Zbiór z Kaggle (20k gier) jest za mały do sensownych korelacji - dobry
najwyżej do wczesnego prototypowania.

Pliki `.pgn.zst` są duże (kilka-kilkanaście GB nieskompresowane) - nie
dekompresujcie ich w całości. `features.py` ma gotową funkcję
`process_pgn_zst_stream()`, która strumieniowo czyta plik `.zst` i
zatrzymuje się po `max_games`, więc możecie np. wziąć pierwsze 20-50k
partii bez ściągania całego miesiąca do pamięci.

## Dokładanie nowych cech

Każda cecha to osobna funkcja w `src/features.py` o sygnaturze:

```python
def feat_something(board_before: chess.Board, move: chess.Move, board_after: chess.Board) -> dict:
    return {"nazwa_cechy": wartość}
```

Dodajcie funkcję i wpiszcie ją do listy `FEATURE_FUNCTIONS` na dole pliku -
pojawi się automatycznie jako kolumna w DataFrame. Tak jak pisał aCat,
jak przyjdą koncepty od Yngviego (pewnie binarne + numeryczne), dokładacie
je dokładnie tym samym wzorcem, bez ruszania reszty pipeline'u.

## Co dalej

1. Uruchom na `sample.pgn`, sprawdź sanity-checki w notebooku.
2. Ściągnij mały wycinek prawdziwych danych (np. 20-50k partii z jednego
   miesiąca Lichess) przez `process_pgn_zst_stream`.
3. Zrób pierwszy przegląd: częstość cech w binach ratingowych, proste
   korelacje (przykłady są w notebooku, sekcja 3).
4. Jak pipeline i wstępne wyniki stoją stabilnie - rozbudowa o kolejne
   cechy i większy zbiór danych.

# TrainedModel – Research-Backtest & Signal-Bot

> **Wichtiger Hinweis:** Dieses Projekt dient ausschließlich **Forschungs- und Bildungszwecken**.
> Es stellt **keine Finanzberatung** dar und sollte ohne umfangreiche zusätzliche Arbeit
> (robuste Validierung, Risikokontrollen, Monitoring und Compliance) nicht für echte
> Handelsentscheidungen verwendet werden.

---

## Überblick

TrainedModel ist ein **research-grade Backtest- und Live-Signal-System** für den Aktienmarkt.
Der Kern besteht aus einem kalibrierten Triple-Barrier-ML-Klassifikator (HistGradientBoosting)
kombiniert mit einer Walk-Forward-Backtest-Engine. Ergänzt wird das System durch einen
Parameter-Sweep-Harness, einen Telegram-Bot für Live-Signale sowie ein modernes
Web-Dashboard zur Analyse der Backtest-Ergebnisse.

---

## Inhaltsverzeichnis

1. [Architektur & Komponenten](#architektur--komponenten)
2. [Voraussetzungen & Installation](#voraussetzungen--installation)
3. [Konfiguration](#konfiguration)
4. [Nutzung – Python-Backtest-Tools](#nutzung--python-backtest-tools)
   - [Einzelner Backtest](#1-einzelner-backtest-backtest_signalspy)
   - [Parameter-Sweep (Grid Search)](#2-parameter-sweep-sweep_backtestspy)
   - [Multi-Ticker OOS-Validierung](#3-multi-ticker-oos-validierung-oos_validationpy)
5. [Nutzung – Web-Dashboard](#nutzung--web-dashboard)
6. [Nutzung – Live-Signal-Bot](#nutzung--live-signal-bot)
7. [Signalmodell-Logik](#signalmodell-logik)
8. [Wichtige Konfigurationsparameter](#wichtige-konfigurationsparameter)
9. [Ausgabe-Struktur](#ausgabe-struktur)
10. [Häufige Fehlerquellen](#häufige-fehlerquellen)
11. [Repo-Hygiene](#repo-hygiene)

---

## Architektur & Komponenten

```
TrainedModel/
├── research/                    # Backtest- & Forschungs-Skripte
│   ├── backtest_signals.py      # Einzelner Walk-Forward-Backtest
│   ├── sweep_backtests.py       # Grid-Search über Hyperparameter
│   └── oos_validation.py        # Multi-Ticker Out-of-Sample-Validierung
│
├── signals/                     # Triple-Barrier Signal-Pipeline
│   ├── config.py                # SignalConfig (alle Hyperparameter)
│   ├── labeling.py              # Triple-Barrier Label-Generierung (-1, 0, +1)
│   ├── model.py                 # HistGradientBoosting + Sigmoid-Kalibrierung
│   ├── engine.py                # FLAT/LONG-Zustandsautomat, EV-basierte Entscheidungen
│   └── dataset.py               # Feature-Matrix-Builder (geflattete Fenster)
│
├── core/                        # Gemeinsame Python-Hilfsfunktionen
│   ├── features.py              # Feature Engineering (BB, Stochastic, OBV, VWAP, CMF, …)
│   ├── storage.py               # SQLite/peewee ORM (Signale, Positionen, Trades)
│   ├── risk.py                  # Kelly-Kriterium, Volatilitäts-Targeting
│   └── sensitivity.py           # Feature-Wichtigkeit & Pruning
│
├── train/                       # Preisprognose-Pipeline (vom Signalmodell getrennt)
│   ├── pipeline.py              # Download → Feature-Engineering → Training
│   ├── core.py                  # Transformer-Ensemble (3-Seed, Dual-Head d1/d5)
│   └── monitor.py               # Modell-Monitoring & Retraining-Trigger
│
├── apps/                        # Full-Stack-Anwendungsschicht
│   ├── api/                     # NestJS 11 REST-API (TypeScript)
│   ├── dashboard/               # React 19 + Vite Frontend (TypeScript)
│   ├── bot_listener.py          # Telegram-Bot-Kommando-Handler
│   ├── jobs.py                  # APScheduler-Job-Definitionen
│   └── main.py                  # Einstiegspunkt für geplante Jobs
│
├── config.example.toml          # Vorlage für Zugangsdaten
├── requirements.txt             # Python-Abhängigkeiten
├── package.json                 # npm-Monorepo (Workspaces: apps/*)
└── docker-compose.yml           # Docker-Konfiguration
```

---

## Voraussetzungen & Installation

### Voraussetzungen

- Python 3.10 oder höher
- Node.js (für Dashboard & API)
- TwelveData API-Key (für OHLCV-Datenabruf)
- Telegram-Bot-Token & Chat-ID (optional, nur für Live-Bot)
- Alpha Vantage API-Key (optional, nur für Sentiment-Feature)

### Python-Umgebung einrichten

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### Node.js-Abhängigkeiten installieren

```bash
npm install
```

---

## Konfiguration

Zugangsdaten aus der Vorlage kopieren:

```bash
# Windows
copy config.example.toml config.toml

# macOS / Linux
cp config.example.toml config.toml
```

Anschließend `config.toml` mit den eigenen Zugangsdaten befüllen:

```toml
[telegram]
bot_token = "DEIN_BOT_TOKEN"
chat_id   = "DEINE_CHAT_ID"

[twelvedata]
api_key = "DEIN_TWELVEDATA_KEY"

[alphavantage]
alva_api_key = "DEIN_ALPHAVANTAGE_KEY"
```

> `config.toml` ist via `.gitignore` ausgeschlossen und darf **niemals** committed werden.

---

## Nutzung – Python-Backtest-Tools

### 1. Einzelner Backtest (`backtest_signals.py`)

Führt einen **Walk-Forward-Backtest** für eine einzelne Aktie durch:

- Zeitreihen-sicheres Walk-Forward-Retraining (alle *N* Bars)
- Konfigurierbares Trainings-Lookback-Fenster
- Einstieg zu `Close[t]` mit realistischer Slippage (nächster Open + Kosten)
- Ausstiege über:
  - **Triple-Barrier** (Stop/Target auf Basis des nächsten Tages OHLC; bei gleichzeitigem Treffer gewinnt Stop)
  - **Zeitlimit** (nach `horizon_days` Haltedauer)
  - **Modellbasierter Ausstieg** (EV-Verschlechterung oder p(Stop) überschreitet Schwelle)

```bash
python research/backtest_signals.py TSLA \
  --start 2018-01-01 --end 2025-12-31 \
  --entry-min-ev 0.20 \
  --retrain-every 80 \
  --lookback 1500 \
  --metrics-start 2024-01-01 --metrics-end 2025-12-31 \
  --tag TSLA_ev020_r80_lb1500
```

**Alle Parameter:**

| Parameter | Standard | Beschreibung |
|---|---|---|
| `ticker` | – | Ticker-Symbol (z. B. `TSLA`) |
| `--start` | – | Backtest-Startdatum (z. B. `2018-01-01`) |
| `--end` | – | Backtest-Enddatum |
| `--entry-min-ev` | `0.12` | Mindest-EV für Einstieg (in R-Einheiten) |
| `--exit-min-ev` | `-0.05` | EV-Schwelle für modellbasierten Ausstieg |
| `--exit-min-p-stop` | `0.55` | Stop-Wahrscheinlichkeits-Schwelle für Ausstieg |
| `--retrain-every` | `20` | Retraining alle N Bars |
| `--lookback` | `2000` | Trainings-Lookback in Bars |
| `--risk` | `0.01` | Risikoanteil pro Trade (1 % des Kapitals) |
| `--initial-cash` | `100000` | Startkapital in USD |
| `--cooldown` | `0` | Wartezeit in Bars nach einem Trade |
| `--no-model-exit` | – | Deaktiviert modellbasierte Ausstiege |
| `--metrics-start` | – | Start des Holdout-Auswertungsfensters |
| `--metrics-end` | – | Ende des Holdout-Auswertungsfensters |
| `--tag` | – | Bezeichnung für den Output-Ordner |
| `--out-dir` | `backtests` | Ausgabeverzeichnis |

**Ausgabe:**

```
backtests/runs/<run_id>/
  trades.csv     Alle Trades mit Entry/Exit, PnL, R-Multiplikator
  equity.csv     Tägliche Kapitalkurve
  meta.json      Run-Metadaten (Config, Git-SHA, Zeitstempel)
```

---

### 2. Parameter-Sweep (`sweep_backtests.py`)

Führt einen **Grid-Search** über mehrere Konfigurationen aus und rankt diese nach einem
wählbaren Zielkriterium. Alle Configs werden im Holdout-Fenster ausgewertet.

```bash
python research/sweep_backtests.py TSLA \
  --start 2018-01-01 --end 2025-12-31 \
  --metrics-start 2024-01-01 --metrics-end 2025-12-31 \
  --entry-min-ev 0.16:0.26:0.02 \
  --retrain-every 60,80,100 \
  --lookback 1200,1500,2000 \
  --objective holdout_sharpe \
  --min-holdout-trades 20 \
  --save-top-k 5 \
  --tag TSLA_grid_2024_2025
```

**Parameter-Spezifikation:**

| Format | Beispiel | Bedeutung |
|---|---|---|
| Kommaliste | `60,80,100` | Genau diese Werte |
| Bereich | `0.16:0.26:0.02` | Start:Stop:Schritt (inklusiv) |

**Ranking-Zielkriterien (`--objective`):**

| Wert | Beschreibung |
|---|---|
| `holdout_sharpe` | Sharpe-Ratio im Holdout-Fenster (Standard) |
| `holdout_return` | Gesamtrendite im Holdout-Fenster |
| `holdout_profit_factor` | Profit-Faktor im Holdout-Fenster |
| `holdout_avg_r` | Durchschnittlicher R-Multiplikator im Holdout |
| `full_sharpe` | Sharpe-Ratio über den gesamten Zeitraum |
| `full_return` | Gesamtrendite über den gesamten Zeitraum |

**Lockbox-Validierung** (einmalige Endvalidierung auf unberührtem Zeitraum):

```bash
python research/sweep_backtests.py TSLA \
  --start 2018-01-01 --end 2024-12-31 \
  --metrics-start 2023-01-01 --metrics-end 2024-06-30 \
  --entry-min-ev 0.16:0.26:0.02 \
  --retrain-every 60,80,100 \
  --lookback 1200,1500,2000 \
  --lockbox-start 2024-07-01 --lockbox-end 2024-12-31 \
  --objective holdout_sharpe \
  --tag TSLA_lockbox
```

**Ausgabe:**

```
backtests/runs/<run_id>/
  sweep_results.csv            Alle Configs + Metriken (sortiert nach Zielkriterium)
  lockbox_validation.json      Lockbox-Ergebnisse der besten Config (optional)
  meta.json
  topk/
    rank01__ev0p200__r80__lb1500/
      trades.csv
      equity.csv
      meta.json
```

---

### 3. Multi-Ticker OOS-Validierung (`oos_validation.py`)

Prüft, ob die gefundene Edge **ticker-übergreifend generalisiert** oder nur für einen
bestimmten Ticker gilt. Führt für eine **feste Config** Backtests auf mehreren Tickers
durch und gibt aggregierte OOS-Metriken aus.

**Modus A – Manuelle Config:**

```bash
python research/oos_validation.py \
  TSLA AAPL NVDA SPY GLD AMD MSFT AMZN \
  --start 2018-01-01 --end 2025-12-31 \
  --holdout-start 2024-01-01 --holdout-end 2025-12-31 \
  --entry-min-ev 0.20 --retrain-every 80 --lookback 1500 \
  --tag oos_grid_best
```

**Modus B – Beste Config automatisch aus Sweep laden:**

```bash
python research/oos_validation.py \
  TSLA AAPL NVDA SPY GLD AMD \
  --start 2018-01-01 --end 2025-12-31 \
  --holdout-start 2024-01-01 --holdout-end 2025-12-31 \
  --from-sweep backtests/runs/<run_id>/sweep_results.csv \
  --tag oos_from_sweep
```

**Interpretationshilfe (wird automatisch ausgegeben):**

| Ergebnis | Bedeutung |
|---|---|
| >70 % der Ticker profitabel | **Stark** – Edge ist ticker-übergreifend |
| 50–70 % der Ticker profitabel | **Moderat** – Edge gilt für die Mehrheit |
| <50 % der Ticker profitabel | **Schwach** – Edge möglicherweise ticker-spezifisch oder überangepasst |

**Ausgabe:**

```
backtests/runs/<run_id>/
  oos_results.csv      OOS-Metriken pro Ticker
  oos_summary.json     Aggregierte Statistiken + verwendete Config
  meta.json
```

---

## Nutzung – Web-Dashboard

Das Dashboard ermöglicht die visuelle Analyse aller Backtest-Ergebnisse.

**Backend starten (Port 3000):**

```bash
cd apps/api
npm run start:dev
```

**Frontend starten (Port 5173):**

```bash
cd apps/dashboard
npm run dev
# Öffnet http://localhost:5173
```

**Verfügbare REST-Endpunkte:**

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/api/runs` | Alle Backtest-Runs auflisten |
| `GET` | `/api/runs/:runId` | Run-Details (geparste CSVs) |
| `GET` | `/api/runs/:runId/download` | CSV herunterladen |
| `GET` | `/api/runs/:runId/topk` | Top-K-Artefakte auflisten |
| `GET` | `/api/runs/:runId/topk/:rank/download/:which` | Spezifisches Ergebnis herunterladen |

---

## Nutzung – Live-Signal-Bot

Der Telegram-Bot ermöglicht tägliche Signal-Abfragen und End-of-Day-Checks.

**Bot starten:**

```bash
python apps/main.py
```

**Telegram-Kommandos:**

| Kommando | Beschreibung |
|---|---|
| `/forecast` | Signal-Job ausführen (BUY / WAIT / HOLD / SELL) |
| `/evaluate` | End-of-Day-Ausstiegsprüfung für offene Positionen |
| `/positions` | Aktuelle offene Positionen anzeigen |
| `/add TICKER` | Ticker zur Watchlist hinzufügen |
| `/remove TICKER` | Ticker von der Watchlist entfernen |
| `/watchlist` | Alle überwachten Ticker anzeigen |

**Job-Ablauf pro Ticker:**

1. Prüfung ob NYSE-Handelstag
2. Neueste OHLCV-Daten herunterladen (mit Caching)
3. Feature Engineering durchführen
4. Signalmodell ggf. nachtrainieren
5. Signal generieren & SQLite-Positionen aktualisieren
6. Ergebnis per Telegram senden

---

## Signalmodell-Logik

### Schritt 1 – Triple-Barrier-Labeling (`signals/labeling.py`)

Für jeden Zeitpunkt *t* wird ein Label generiert:

```
Einstiegspreis  = Close[t]
Stop-Preis      = Einstieg − stop_loss_atr × ATR
Target-Preis    = Einstieg + take_profit_atr × ATR
Vorausschau     = horizon_days Handelstage

Label = +1  wenn Target zuerst erreicht  (Gewinn)
Label = -1  wenn Stop zuerst erreicht    (Verlust)
Label =  0  wenn keines erreicht         (Timeout)
```

### Schritt 2 – Modelltraining (`signals/model.py`)

```
Features   : Geflattete Schiebefenster der Breite window_size (Standard: 60)
Modell     : HistGradientBoostingClassifier (scikit-learn)
Gewichtung : sqrt_inverse_frequency (gedämpftes Klassen-Balancing)
Kalibrierung: Sigmoid-Methode auf den letzten 20 % der Daten (zeitreihen-sicher)
Ausgabe    : p(-1), p(0), p(+1) – kalibrierte Wahrscheinlichkeiten
```

### Schritt 3 – EV-basierte Entscheidung (`signals/engine.py`)

```
R        = take_profit_atr / stop_loss_atr   (Reward/Risk-Verhältnis)
Kosten_R = Roundtrip-Kosten / Stop-Abstand  (in R-Einheiten)
EV_net   = p(+1) × R − p(−1) − Kosten_R

Einstieg : EV_net ≥ entry_min_ev
Ausstieg : EV_net ≤ exit_min_ev  ODER  p(Stop) ≥ exit_min_p_stop
```

### Positionsgrößenbestimmung

```
Risikobudget = Kapital × risk_per_trade
Aktien       = Risikobudget / (Einstieg − Stop)
Begrenzt auf: max_position_fraction × Kapital
```

---

## Wichtige Konfigurationsparameter

Alle Hyperparameter befinden sich in `signals/config.py` als `SignalConfig`-Dataclass.

| Parameter | Standard | Beschreibung & Effekt |
|---|---|---|
| `window_size` | `60` | Länge des Feature-Fensters in Bars |
| `horizon_days` | `10` | Triple-Barrier-Horizont in Handelstagen |
| `take_profit_atr` | `1.5` | Target-Abstand in ATR-Einheiten |
| `stop_loss_atr` | `1.0` | Stop-Abstand in ATR-Einheiten |
| `entry_min_ev` | `0.12` | **Höher** → weniger, aber bessere Trades; Risiko: zu wenige OOS-Trades |
| `exit_min_ev` | `-0.05` | EV-Schwelle für modellbasierten Ausstieg |
| `exit_min_p_stop` | `0.55` | Stop-Wahrscheinlichkeits-Schwelle für frühzeitigen Ausstieg |
| `retrain_every` | `20` | **Kleiner** → reaktiver, mehr Rechenaufwand; **Größer** → stabiler |
| `lookback` | `2000` | **Größer** → stabiler; **Kleiner** → adaptiver an aktuelle Marktregime |
| `one_way_cost_bps` | `5.0` | Einseitige Transaktionskosten in Basispunkten (Provision + Spread) |
| `calibration_ratio` | `0.20` | Anteil der Daten für Kalibrierung (letzte 20 %) |
| `max_class_weight` | `3.0` | Maximales Klassen-Gewicht (verhindert Übergewichtung seltener Klassen) |

---

## Ausgabe-Struktur

Alle Backtest-Ergebnisse werden in `backtests/runs/` abgelegt:

```
backtests/
  runs/
    <TIMESTAMP>__<TICKER>__<TAG>__backtest/    ← Einzelner Backtest
      trades.csv        Trades (Entry, Exit, PnL, R-Multiplikator, Ausstiegsgrund)
      equity.csv        Tägliche Kapitalkurve
      meta.json         Config, Git-SHA, Plattform, CLI-Parameter

    <TIMESTAMP>__<TICKER>__<TAG>__sweep/       ← Parameter-Sweep
      sweep_results.csv             Alle Configs + Full/Holdout-Metriken
      lockbox_validation.json       Lockbox-Ergebnisse (optional)
      meta.json
      topk/
        rank01__ev0p200__r80__lb1500/
          trades.csv
          equity.csv
          meta.json

    <TIMESTAMP>__MULTI__<TAG>__oos/            ← OOS-Validierung
      oos_results.csv     OOS-Metriken pro Ticker
      oos_summary.json    Aggregat-Statistiken
      meta.json
```

---

## Häufige Fehlerquellen

**Überanpassung durch Parameter-Sweeps**
Die beste Config im Holdout ist kein echter Out-of-Sample-Test mehr. Sweep = Modellselektion,
danach auf einem *frischen* Zeitraum oder anderen Tickers validieren (→ `oos_validation.py`).

**Kleine-Stichproben-Illusion**
Sehr hoher Sharpe oder Profit-Faktor bei wenigen Trades ist statistisch nicht belastbar.
`--min-holdout-trades` in Sweeps konsequent nutzen.

**Unadjustierte OHLCV-Daten**
Manche TwelveData-Versionen liefern keine adjustierten Kurse. Das System warnt,
fällt aber auf unadjustierte Daten zurück – besonders kritisch bei Splits.

**Tickerspezifische Edge**
Immer mit `oos_validation.py` auf mindestens 5–8 verschiedenen Tickers prüfen,
ob die Strategie generalisiert.

**Zugangsdaten im Repository**
`config.toml` enthält API-Schlüssel und darf **niemals** committed werden.
`.gitignore` schützt die Datei – jedoch nie `git add -f` verwenden.

---

## Repo-Hygiene

- Generierte Artefakte (Backtest-CSVs, Cache, SQLite-DB, Modelle) sind via `.gitignore` ausgeschlossen.
- `backtests/`, `cache/`, `models/`, `models_signals/` und `logs/` sind Ausgabeverzeichnisse und bleiben im Source-Control leer.
- `config.toml` niemals committen.

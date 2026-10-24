# Sign Language to Text Converter

This repository contains a Python-based workflow for collecting, training, and deploying an American Sign Language (ASL) alphabet recognizer backed by a simple DBMS (SQLite). The workflow is split into three stages:

1. **Dataset capture** – collect labeled hand-landmark samples for every alphabet letter using a webcam and MediaPipe Hands.
2. **Model training** – train a scikit-learn classifier on the captured dataset and export it as a `.joblib` file.
3. **Runtime app** – run a Tkinter/OpenCV application that converts live signs into text and stores every recognized character plus finalized sentences inside a SQLite database.

## Project structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── samples.csv          # created after running the collector
├── models/
│   └── alphabet_clf.joblib  # created after training
├── app/
│   ├── collect_alphabet.py  # data collection GUI
│   ├── train_alphabet.py    # training pipeline
│   ├── recognize.py         # real-time inference UI (todo)
│   └── storage.py           # SQLite helper (todo)
```

All three stages are available; start by collecting data, then train, then run the recognizer.

## 1. Set up the environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Collect alphabet samples

Run the collector and press the letter key (`A-Z`) that matches the sign you are performing. Each key press records one landmark vector into `data/samples.csv`.

```bash
python app/collect_alphabet.py --output data/samples.csv
```

Tips:
- Use consistent lighting and framing.
- Capture at least ~200 samples per class for a dependable model.
- Use `--mirror` if you want the preview to mimic a mirror.

## 3. Train (or rebuild) the classifier

You can **skip this step** because a pretrained model is already included (`models/alphabet_clf.joblib` produced from `data/pretrained_samples.csv`, which augments the seed samples with noise). If you capture fresh data or want to regenerate the pretrained artifacts:

```bash
# Option A: train directly on your collected CSV
python app/train_alphabet.py --input data/samples.csv --model models/alphabet_clf.joblib

# Option B: rebuild the synthetic pretrained set + model
python app/build_pretrained.py \
    --input data/samples.csv \
    --output data/pretrained_samples.csv \
    --model models/alphabet_clf.joblib
```

The builder script creates a much larger dataset via Gaussian augmentation (default 400 samples per class) and trains a model automatically.

## 4. Run the recognizer with DB storage

```bash
python app/recognize.py --mirror
```

Features:
- Live webcam preview with MediaPipe landmarks and current prediction.
- Tkinter UI that accumulates letters, supports Space/Delete/Clear, and saves sentences.
- Every committed letter is logged to `data/sign_texts.db` (`letter_events` table), and clicking *Save to DB* stores the whole sentence plus a per-letter breakdown.
- Optional AI assistance: press *Grammar Fix* to run LanguageTool over the sentence, review the suggestion in the lower pane, and click *Apply Suggestion* to copy it back before saving. (Requires Java 8+ for `language-tool-python`.)

Use `Esc` in the OpenCV window or the Quit button to exit. Every letter is logged to `letter_events`; saving the sentence inserts it and expands individual characters into `sentence_letters`.

## 5. Next steps

- Collect more samples for better accuracy and retrain periodically.
- Add tests plus CI hooks (pre-commit/ruff) once the core functionality stabilizes.
- Package pretrained artifacts + Java instructions if distributing to other machines.

## Troubleshooting

- **MediaPipe install errors** – make sure you are using Python 3.9–3.11 on Windows. Older versions may fail to compile dependencies.
- **No camera** – use `--video path/to/file.mp4` to collect from prerecorded footage.
- **Class imbalance** – retrain after collecting more samples for underrepresented letters.


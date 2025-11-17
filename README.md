Apple Vision – MinneApple Starter

Überblick
- Ziel: Einfache Trainings-Pipeline zur Erkennung von Äpfeln in Bildern.
- Datensatz: MinneApple (Detection). Dieses Repo nutzt Faster R-CNN (torchvision) auf COCO‑ähnlichen Annotationen (nur Bounding Boxes).

Voraussetzungen
- Python >= 3.10
- Abhängigkeiten (siehe pyproject.toml): torch, torchvision, pycocotools, pillow, numpy, tqdm, ruff
- Optional: CUDA-fähige GPU (Training auf CPU ist möglich, aber deutlich langsamer).

Installation
- Mit uv (empfohlen):
  - uv sync
  - Ausführen ohne manuelles Aktivieren der Umgebung: uv run python -m apple_vision.train --help
- Mit pip (Alternative):
  - python -m venv .venv
  - .venv\Scripts\activate  (Windows) oder source .venv/bin/activate (Linux/macOS)
  - pip install -e .

Datensatz vorbereiten (MinneApple)
- Ausgangsdaten (Beispiel): data/minneapple/detection mit Unterordnern train/, test/ und jeweils images/ und masks/.
- Konvertiere nach COCO-Bounding-Box-Format mit:
  - uv run python scripts/prepare_minneapple.py --root data/minneapple/detection --val-ratio 0.15 --seed 42
- Ergebnisstruktur (COCO):
  - data/minneapple/coco/
    - images/
      - train/
      - val/
    - annotations/
      - instances_train.json
      - instances_val.json
      - [instances_test.json]

Datensatz vorbereiten (Apple Dataset Benchmark from Orchard Environment)
- Download der Rohdaten (Supervisely-Archiv oder Original-ZIPs) z. B. über Dataset Ninja: https://datasetninja.com/apple-dataset-benchmark-from-orchard-environment
- Nach dem Entpacken sollte ein Ordner mit Unterordnern ArtificialLight/, CropLoadEstimation/, HarvestingRobot2016/, HarvestingRobot2017/ vorliegen (Standard: data/orchard/raw).
- Konvertiere nach COCO-Bounding-Box-Format mit:
  - uv run python scripts/prepare_orchard_benchmark.py --root data/orchard/raw --val-ratio 0.15 --seed 42
  - Optional: --test-ratio <0.x> für einen dritten Split und --symlink zum Symlinken statt Kopieren.
- Ergebnisstruktur (COCO):
  - data/orchard/coco/
    - images/train, images/val [, images/test]
    - annotations/instances_train.json, annotations/instances_val.json [, instances_test.json]
  - Train/Val/Test sind Zufallssplits über alle 2.299 Bilder (Reproduzierbarkeit via --seed).

Schnellstart: Training
- Beispiel (gerät wird automatisch gewählt – CUDA falls verfügbar):
  - uv run python -m apple_vision.train --dataset-root data/minneapple/coco
- Alternativ ohne uv:
  - python -m apple_vision.train --dataset-root data/minneapple/coco
- Auch möglich (delegiert an das Training-CLI):
  - python main.py --dataset-root data/minneapple/coco

Wichtige Optionen (mit Defaults)
- --train-ann (annotations/instances_train.json)
- --train-images (images/train)
- --val-ann (annotations/instances_val.json)
- --val-images (images/val)
- --epochs (1)
- --batch-size (2)
- --lr (0.005)
- --num-workers (2)
- --out-dir (checkpoints)
- --resume <pfad-zu-checkpoint.pth> (optional)
- --early-stop-patience <N> (0 = deaktiviert)

Ausgaben/Checkpoints
- Bestes Modell (nach Val-Loss): checkpoints/fasterrcnn_resnet50_fpn_apple_best.pth
- Finales Modell (am Trainingsende): checkpoints/fasterrcnn_resnet50_fpn_apple.pth

Hinweise
- Dieses Starterprojekt verwendet ausschließlich Bounding Boxes; eine spätere Erweiterung auf Mask R-CNN ist möglich.
- Falls dein Rohdatensatz nicht im erwarteten Format vorliegt, nutze das Skript scripts/prepare_minneapple.py oder konvertiere zu COCO (bbox=[x,y,w,h], category_id=1 für "apple").

Nächste Schritte (Roadmap)
- Maskenunterstützung (Mask R-CNN)
- REST API (FastAPI) für Inferenz

Datenvalidierung/Visualisierung – Schnellplot
- Schneller visueller Check deiner Annotationen (zeichnet Ground‑Truth-Bounding‑Boxes auf Beispielbilder).
- Beispiel:
  - uv run python -m apple_vision.quickplot --dataset-root data/minneapple/coco --ann annotations/instances_val.json --images images/val --n 8 --out-dir quickplots
- Optional: --show öffnet die Bilder nach dem Speichern.

Metriken – mAP (COCO Evaluator)
- Evaluiert ein gespeichertes Modell gegen das Val‑Set (COCO‑Style) und reportet mAP (AP@[.5:.95]) sowie AP50/AP75 usw.
- Beispiel:
  - uv run python -m apple_vision.evaluate_coco --dataset-root data/minneapple/coco --val-ann annotations/instances_val.json --val-images images/val --checkpoint checkpoints/fasterrcnn_resnet50_fpn_apple_best.pth
- Ergebnis: COCO‑Summary in der Konsole und Speicherung der Detections als JSON (coco_results.json).

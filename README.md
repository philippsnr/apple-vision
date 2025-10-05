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
- Datenvalidierung/Visualisierung (Schnellplot mit Bounding Boxes)
- Metriken (mAP, COCO Evaluator)
- Maskenunterstützung (Mask R-CNN)
- REST API (FastAPI) für Inferenz

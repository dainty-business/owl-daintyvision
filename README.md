# DaintyVision: time to biochemical recurrence from one prostatectomy H&E slide (OWL 2026)

This is team DaintyVision's code for the OWL challenge on Grand Challenge. From one whole-slide image of a radical
prostatectomy, the algorithm predicts the time to biochemical recurrence (BCR). It also writes a risk class, an
attribution map of the slide, and a PDF report. It uses no Gleason/ISUP grade, staging, annotation or biomarker at any
step.

## How it works

1. **Tiles.** The slide is cut into 224 x 224 px tiles at about 0.97 um/px, keeping tiles with at least 50% tissue in
   the provided tissue mask. A slide without a level near 0.97 um/px is resampled from a finer level. Tiles that are
   more than 90% near-white are dropped. Large slides use a fixed, reproducible set of at most 28000 tiles (the cells
   with the lowest fixed random key), so repeated runs see the same tiles.
2. **Features.** Each tile is encoded by the frozen Kaiko Midnight encoder (ViT-g/14): [CLS, mean of patch tokens],
   3072 dimensions.
3. **Slide model.** An additive gated attention MIL head, trained with the Cox partial likelihood on the 508 training
   patients. Ten members are trained with different seeds, and their risks are averaged. The slide risk of one member
   is R = sum_i a_i r_i, a softmax-weighted mean of tile risks.
4. **Outputs.** The predicted time is the restricted mean BCR-free time over 10 years from the Breslow baseline. The
   risk class compares that time with the thirds of the training cohort's predicted times.
5. **Attribution map.** Removing tile i changes a member's R by exactly -a_i (r_i - R) / (1 - a_i). The map ranks tiles
   by this effect, averaged over members, in the direction (risk-raising or risk-lowering first) whose removal moves
   the slide's prediction more. A per-slide removal test decides the direction. Map values are rank^4, as uint8.
6. **Report.** The five-page PDF report opens with a case summary: the tissue behind the score, the prediction when
   the model is re-run without the 10% most risk-raising or risk-lowering tiles, ensemble agreement, distance to the
   class cut-off and case-specific cautions. It then shows the ranked and signed maps, the 16 most influential tiles,
   a page on recurring tissue patterns, removal curves for both directions against random removal, and about 1 mm of
   context around the strongest tiles with a map of artefact patterns. The pattern
   page uses 16 clusters of tile features from all training slides, formed without outcome or annotations. Their
   outcome statistics are computed afterwards on the training cohort and on TCGA-PRAD. The pattern names were written
   from exemplar tiles by an AI assistant and are not verified by a pathologist.

## Layout

- `container/`: the inference container (Grand Challenge invoke API, `app.py`), the model code (`inference.py`,
  `owl_wsi.py` tiling and encoding, `owl_mil.py` MIL head, `report.py`), `Dockerfile`, and `Dockerfile.selfcontained`
  with a test script for a single image that includes the model.
- `tools/`: feature extraction, training, external evaluation, the tissue-pattern atlas, model packaging, the
  faithfulness replica and checks.

## Weights and data

No weights are included.

- **Encoder.** Kaiko Midnight (MIT licence): https://huggingface.co/kaiko-ai/midnight. Save an fp16
  `save_pretrained` copy (config.json, model.safetensors) to a folder.
- **Training data.** The OWL training slides, tissue masks and labels (the LEOPARD cohort), as the challenge
  distributes them. `tools/extract_features.py` streams the slides from the challenge's public shares.
- **External data.** The TCGA-PRAD BCR dataset curated by the organizers: https://huggingface.co/datasets/waticlems/tcga-prad-bcr.
  It serves only as an external validation cohort: its slides are encoded (`tools/extract_tcga.py`) and scored with
  its BCR labels, and it is never trained on. Midnight-12K itself was pretrained without labels on 12k TCGA slides,
  which likely include these.

## Reproducing the model

Each tool's docstring gives its full usage. Paths below are examples.

```bash
python tools/extract_features.py --out work --weights weights/midnight
python tools/extract_tcga.py --out work/tcga --weights weights/midnight
python tools/train_mil.py --feats work/feats --labels training_labels.csv --out runs/cv --epochs 10 --lr 5e-5
python tools/train_mil.py --feats work/feats --labels training_labels.csv --out runs/final --final --seeds 10 --epochs 10 --lr 5e-5
python tools/eval_external.py --train-feats work/feats --train-labels training_labels.csv --ext-feats work/tcga/feats --ext-labels work/tcga/labels.csv --model-dir runs/final --out runs/tcga.json
python tools/build_atlas.py --model-dir runs/final --feats work/feats --labels training_labels.csv --ext-feats work/tcga/feats --ext-labels work/tcga/labels.csv --gallery-raw slides --masks work/masks --out atlas
python tools/pack_model.py runs/final container model.tar.gz midnight=weights/midnight --atlas atlas
```

The first `train_mil.py` call runs the cross-validation; the second trains the 10 final members (seeds 1000-1009).
`build_atlas.py` needs a few full training slides for its exemplar galleries (`tools/fetch_slides.py`). Write the
pattern names to `atlas/atlas_descriptions.json` before packing, as
`{"clusters": {"0": {"name": ..., "group": ..., "description": ...}, ...}}`, with group "artefact" for artefact
patterns.

## Running inference

```bash
cd container && docker build --platform=linux/amd64 -t owl_daintyvision .
```

Extract `model.tar.gz` to a folder and mount it at `/opt/ml/model`. Put one slide in
`/input/images/prostatectomy-wsi/` and its mask in `/input/images/prostatectomy-tissue-mask/`. The container serves
Grand Challenge's invoke API on port 4743. `inference.get_args_parser()` and `inference.run(args)` serve the
organizers' faithfulness notebook. A Tesla T4 encodes 32-35 tiles/s, so a slide takes about 7-15 minutes.

## Licence

See `LICENSE`. Kaiko Midnight has its own licence (MIT).

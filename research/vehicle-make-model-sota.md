# Vehicle Make & Model Detection — SOTA Research

**Date:** 2026-06-16
**Context:** AI-Traffic-Analysis (Vietnamese roads).
Current YOLO: 5 coarse classes (bicycle/bus/car/motorbike/truck), zero make/model capability.

---

## Current Project Audit

- **Architecture:** Two-stage YOLO (YOLOv9s vehicle detector → plate detector → PaddleOCR)
- **Has make/model:** NO
- **Vehicle classes:** bicycle, bus, car, motorbike, truck (coarse only)
- **Weight files:** vehicle_yolov9s_640, plate yolo8n/9s/11n/12n, DeepSORT ckpt
- **Gaps:**
  - No make/model classifier, no dedicated weights, no inference call
  - `Vehicle` dataclass has no `make`/`model` fields
  - `VehicleDetection` API exposes only `vehicle_type`, `license_plate`, `bbox`, `confidence`
  - `VEHICLES` constant maps 5 COCO indices only

---

## Ranked Repos (with pretrained weights)

| Score | Name                                                 | Arch            | Weights       | Notes                                                |
| ----- | ---------------------------------------------------- | --------------- | ------------- | ---------------------------------------------------- |
| 7.75  | [dima806/car_models_image_detection][hf-d806]        | ViT-B16         | HF Hub        | 84.1% acc; MIT; active 2024; transformers dep        |
| 7.5   | [Jordo23/vehicle-classifier][hf-j23]                 | EfficientNet-B4 | .pth + .onnx  | 8,949 VMMRdb classes; MIT; **ONNX fits onnxruntime** |
| 6.5   | [kamwoh/Car-Model-Classification][gh-kmw]            | ResNeXt50 V2    | in-repo .pth  | ~94% top-1 on Cars-196; stale 2022                   |
| 6.25  | [foamliu/Car-Recognition][gh-far]                    | ResNet-152      | release .hdf5 | Make-only (163 classes); MIT; stale 2022             |
| 4.75  | [Pells31/Vehicle-Make-and-Model-Recognition][gh-p31] | ConvNet         | None          | Active 2025; supports Stanford Cars + VMMRdb         |
| 3.5   | [Helias/Car-Model-Recognition][gh-hel]               | ResNet-152      | None          | Clean code; Cars-196; stale 2022                     |

[hf-d806]: https://huggingface.co/dima806/car_models_image_detection
[hf-j23]: https://huggingface.co/Jordo23/vehicle-classifier
[gh-kmw]: https://github.com/kamwoh/Car-Model-Classification
[gh-far]: https://github.com/foamliu/Car-Recognition
[gh-p31]: https://github.com/Pells31/Vehicle-Make-and-Model-Recognition
[gh-hel]: https://github.com/Helias/Car-Model-Recognition

> Score = avg(accuracy + has_weights + maintenance + integration) / 4, LLM-assigned. Accuracy UNVERIFIED.

---

## Recommended Top Pick: Jordo23/vehicle-classifier

**Why:** ONNX weights fit directly into existing `onnxruntime` sessions — no new runtime.
EfficientNet-B4 light enough for M1/16GB CPU on already-cropped vehicle images. MIT license.

**Integration plan (~3 days):**

1. Add `make: str`, `model: str` to `Vehicle` dataclass (`utils/utils.py:51`)
2. Load Jordo23 `.onnx` session alongside existing vehicle/plate sessions
3. Insert classify-on-crop call after vehicle match at `traffic_analysis.py:273`,
   feeding `v.vehicle_image`; gate to `type in {car, truck, bus}`
4. Extend `VehicleDetection` schema (`main.py:65`) and response builders
   (`main.py:141/202`) with make/model fields
5. Run VN sanity set — expect poor accuracy on VN-market trims (fine-tuning needed)

**Runner-up:** dima806/car_models_image_detection — higher score (7.75), trivial HF
`from_pretrained`, ViT-based; demoted: requires `transformers` dep vs onnxruntime.

---

## Ranked Papers

| Score | Paper                                                        | Year | Acc        | Code             | Weights                  |
| ----- | ------------------------------------------------------------ | ---- | ---------- | ---------------- | ------------------------ |
| 9     | [PMAL/PMD — Anti-Noise Fine-Grained Vehicle][pmal]           | 2024 | SOTA Cars  | Yes (Apache-2.0) | TResNet-L on InfiniCLOUD |
| 8     | [Two-Stage RT-DETR + ViT Pipeline][rtdetr]                   | 2026 | —          | Yes              | —                        |
| 7.5   | [TransFG: Transformer for Fine-Grained Recognition][transfg] | 2021 | 92.3% Cars | Yes              | Available                |
| 7     | [Multi-Task Hierarchical Make+Model][multitask]              | 2026 | —          | No               | —                        |
| 7     | [UFPR-VeSV: Unified Fine-Grained Vehicle + ALPR][ufpr]       | 2026 | —          | Yes              | —                        |

[pmal]: https://arxiv.org/abs/2401.14336
[rtdetr]: https://arxiv.org/abs/2606.05149
[transfg]: https://arxiv.org/abs/2103.07976
[multitask]: https://arxiv.org/abs/2603.01746
[ufpr]: https://arxiv.org/abs/2604.05271

### Best scratch implementation: PMAL/PMD (arXiv:2401.14336)

- Anti-noise training = robust on degraded traffic-cam crops (blur, occlusion, night, far-field)
- Evaluates across 5 datasets → recipe transfers to custom CompCars+VN training set
- **~3 weeks effort** (port + fine-tune on VN data; excludes data collection)

---

## Critical Warning: Vietnam Context

> **Every repo and dataset above is CAR-ONLY.**

Vietnamese roads are motorbike-dominant. 2 of 3 examples (Honda Wave, Yamaha Exciter) are
motorbikes. No public dataset covers VN motorbike make/model.

### VN-specific issues:

1. **Motorbikes are the real target** — car classifiers solve the minority use case
2. **Brand distribution is Asian-specific:** Honda/Yamaha/Suzuki/Piaggio (bikes);
   Toyota/Hyundai/Kia/Mitsubishi/**VinFast** (cars) — VinFast absent from all Western datasets
3. **Gate by YOLO type** — run motorbike head on `type==motorbike`, car head on `car/truck/bus`;
   single car-trained head on bikes = silent garbage
4. **Leverage plate-keyed pseudo-labeling** — existing pipeline crops `vehicle_image` and reads
   plates; group crops by plate for cheap VN labeling bootstrap
5. **Confidence-based abstention** — threshold ~0.60; unknown VN trims should abstain

### Recommended dataset path:

1. **CompCars** — closest Asian-market car dataset; Honda/Toyota/Hyundai/Kia overlap
2. **Custom VN collection** (mandatory) — plate-keyed crop extraction from existing pipeline
3. **UFPR-VeSV** — Brazilian real-surveillance CCTV; useful for benchmark methodology

---

## Next Steps (Prioritized)

- [ ] Wire Jordo23 ONNX for car branch — `type in {car,truck,bus}` only — 3 days
- [ ] Build VN motorcycle taxonomy (top-10 models by registration) — Week 1
- [ ] Collect VN motorbike crops via existing pipeline, plate-keyed labeling — Weeks 2-4
- [ ] Fine-tune EfficientNet-B3 on VN motorcycle data — Weeks 4-6
- [ ] Fine-tune PMAL/PMD on CompCars + custom VN car set (parallel track)

---

# Motorcycle / Motorbike Make & Model — SOTA Research

**Date:** 2026-06-16
**Verdict: No turnkey solution for VN models. One repo has two-wheeler weights (Indian brands).
Custom data collection mandatory.**

---

## Gap Assessment

5-7 years behind car recognition. Cars have VMMRdb (291K imgs / 9,170 classes) + multiple
downloadable weight repos. Motorcycles have:

- One peer-reviewed dedicated study (Lima 2025, SIBGRAPI Workshop-in-Progress)
- One repo with real downloadable two-wheeler weights (FGVD — Indian brands, not VN)
- No public source covers Vietnamese-market models at scale
- VinFast (Klara/Evo): zero public data, unrecoverable gap, must self-collect

---

## Ranked Repos

| Score | Repo                                                 | Weights      | Domain           | Two-Wheeler Coverage                     |
| ----- | ---------------------------------------------------- | ------------ | ---------------- | ---------------------------------------- |
| 6.5   | [iHubData-Mobility/public-FGVD][fgvd]                | Yes (Zenodo) | Indian roads     | Honda/Yamaha/Hero/Bajaj/TVS; L-3 mAP 48% |
| 5.25  | [Lima001/UFPR-FGMC][fgmc]                            | No           | Brazilian toll   | Honda/Yamaha/Suzuki/Kawasaki; 95% make   |
| 4.5   | [Pells31/Vehicle-Make-and-Model-Recognition][gh-p31] | Yes (.pt)    | —                | Cars only; good timm scaffold            |
| 2.75  | [leogodin217/motorcycle_classification][gh-leo]      | No           | Western catalog  | 366 classes but ~7.6 imgs/class          |
| 1.75  | [anthonybaulo/moto-classifier][gh-ant]               | No           | Western cruisers | 6 makes; 55.3% top-1; GPL-3.0            |

[fgvd]: https://github.com/iHubData-Mobility/public-FGVD
[fgmc]: https://github.com/Lima001/UFPR-FGMC
[gh-leo]: https://github.com/leogodin217/motorcycle_classification
[gh-ant]: https://github.com/anthonybaulo/moto-classifier

**Top pick: FGVD** — only repo with real downloadable two-wheeler weights (BSD-3-Clause).
Indian-market models need retraining for VN, but pipeline + recipe is the best starting point.

**Caveats on FGVD:**

- Weights on [Zenodo 7499479](https://zenodo.org/record/7499479): YOLOv5L + HRN
- Indian models (Hero Splendor, Bajaj Pulsar) absent from VN roads → classifier head needs full retrain
- HRN head is ONNX-hostile; swap to EfficientNet-V2 for this project's onnxruntime constraint
- Zenodo dataset license unconfirmed variant (may be CC-BY-NC) — verify before commercial use

---

## Ranked Datasets

| Score | Dataset                                | Size                      | VN Models                           | License         | Access        |
| ----- | -------------------------------------- | ------------------------- | ----------------------------------- | --------------- | ------------- |
| Best  | Thai Motorcycle (Roboflow / Satayu C.) | ~1.1K imgs                | Wave, PCX, Exciter, Aerox/NVX       | CC BY 4.0       | Public        |
| 2     | FGVD (IIIT Hyderabad / Zenodo)         | 5,502 imgs / 24,450 boxes | Indian brands (no VN models)        | CC (unverified) | Public        |
| 3     | UFPR-VeSV (arXiv:2604.05271)           | 24,945 imgs               | Brazilian Honda/Yamaha              | Research        | Email request |
| 4     | Honda Wave/PCX (Roboflow)              | 226 imgs                  | Wave 110/125, PCX                   | CC BY 4.0       | Public        |
| 5     | RodoSol-ALPR moto subset               | ~10K imgs                 | Brazilian; no make/model labels yet | Research        | Email request |

> Thai Roboflow = only public source with actual VN-market models. Thin (seed-only, ~1.1K imgs).
> VinFast Klara/Evo: ZERO public data. Self-collect mandatory.

**Make/model coverage split:**

- **Make-level** (Honda/Yamaha/Suzuki): broad overlap Thai + FGVD + UFPR-VeSV
- **Model-level VN**: only Thai Roboflow + Honda Wave Roboflow cover VN-specific models
- **VinFast**: unrecoverable gap — no public data anywhere

---

## Ranked Papers

| Score | Paper                                                      | Year | Make Acc    | Code | Weights |
| ----- | ---------------------------------------------------------- | ---- | ----------- | ---- | ------- |
| 9     | Fine-Grained Motorcycle Classification for ITS (Lima/UFPR) | 2025 | 95.4% micro | Yes  | No      |
| 8     | Transfer Learning for Two-Wheeler Brand+Model (Indian)     | 2024 | AP 0.9435   | No   | No      |
| 7     | FGVD: Fine-Grained Vehicle Detection Unconstrained Roads   | 2022 | L-2 mAP 59% | Yes  | Yes     |
| 6     | UFPR-VeSV: Unified Fine-Grained Vehicle + ALPR             | 2026 | —           | Yes  | No      |
| 5     | MoRe: Large-Scale Motorcycle Re-ID (WACV 2021)             | 2021 | N/A (re-ID) | No   | No      |

**Best implementation recipe: Lima 2025 (SIBGRAPI)**

- EfficientNet-V2, timm-based → ONNX-friendly, fits onnxruntime
- Make micro-acc 95.4% / macro 77.1%; Model micro-acc 94.6% / macro 85.5%
- Macro gap (77% vs 95%) = long-tail collapse warning — plan for it
- No weights; Brazilian data; but architecture + training recipe is directly reproducible

---

## VN-Specific Path (Revised)

### Week 1 — Taxonomy + FGVD Baseline

- Define VN taxonomy: ~15-25 top models
  - Honda: Wave Alpha/RSX, Vision, Air Blade, SH150i/Mode, PCX, Winner
  - Yamaha: Exciter 150/155, NVX 155, Sirius, Janus, Grande
  - Suzuki: Raider R150
  - Piaggio: Vespa Primavera/Sprint
  - VinFast: Klara, Evo (self-collect only)
- Pull FGVD repo + Zenodo weights; reproduce L-1/L-2/L-3 baseline to confirm pipeline

### Weeks 2-4 — Data Collection (dominant cost)

- Mine VN motorbike crops from existing `motorbike` detector output (`vehicle_image`)
- Seed labels from Thai Roboflow (Exciter/Aerox/Wave/PCX) + Honda Wave Roboflow
- Manually annotate VN-specific models absent from all public sets
- Target ≥300-500 labeled crops/class; plate-keyed grouping for identity dedup

### Week 2 (parallel) — Model Swap

- Replace FGVD's HRN head with EfficientNet-V2 (Lima recipe) or EfficientNet-B4 (Jordo23)
- Both export to ONNX cleanly via timm
- Pretrain on FGVD + Thai Roboflow before VN fine-tune

### Weeks 5-8 — VN Fine-tune + Integration

- Fine-tune on collected VN crops; monitor macro-F1 (not just micro-accuracy)
- Confidence abstention ~0.65 for unknown models
- Export to ONNX; gate to `type == motorbike`; insert after existing detector
- Add `make`/`model` to `Vehicle` dataclass + `VehicleDetection` API

**Total: 6-9 weeks. Start make-level (fast, ~95% micro); expand to model-level as data grows.**

---

## Asian Brand Coverage — Honest Assessment

| Source              | VN Make                         | VN Model                         | Notes                             |
| ------------------- | ------------------------------- | -------------------------------- | --------------------------------- |
| Thai Moto Roboflow  | Honda, Yamaha                   | Wave, PCX, Exciter, Aerox/NVX    | Only direct VN-model match. Thin. |
| Honda Wave Roboflow | Honda                           | Wave 110/125, PCX                | Honda-only, 226 imgs              |
| FGVD                | Honda, Yamaha                   | Indian models (Activa, Splendor) | Makes transfer; models don't      |
| UFPR-FGMC           | Honda, Yamaha, Suzuki, Kawasaki | Brazilian (CG, Factor)           | Makes transfer; models don't      |
| VinFast             | —                               | —                                | ZERO public data. Self-collect.   |

> Yamaha Aerox = NVX in VN (same platform). Honda Click ≈ Vision (shared underbone, not 1:1).
> Hero/Bajaj/TVS (FGVD Indian brands) = absent from VN roads entirely.

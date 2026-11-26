# CSP650 photo-only model-family pilot
The original Roboflow exports supplied bounding boxes and brand labels for the detector. The new model-family labels come from the folder names of the published Google Lens watch experiment, pinned at commit 33cd75330c0e7a7cf0796a06543f25deb7f3fe87. A reproducible seeded split, after filtering exact and near-duplicate perceptual hashes, yielded 118 gallery and 40 test photographs. There were eight test photos per family. The source photos are product images from one third-party collection. Its labels were used as supplied, and hash filtering cannot exclude all alternate views of the same physical watch.

Each prediction used the test image alone. A pretrained ResNet-18 encoded the full photo. The app ranked each family by the mean cosine similarity of its three closest gallery photos, with a fixed bonus for a distinctive model word recognized by OCR. YOLOv8 Nano supplied an OCR crop when it returned a box. The detector's brand label never restricted the gallery. Only gallery photos were indexed. No family, brand or reference was entered during tests.

| Family | Test photos | Correct first | Correct in first three |
|---|---:|---:|---:|
| Hublot Big Bang | 8 | 6 | 8 |
| Jaeger-LeCoultre Reverso | 8 | 8 | 8 |
| Rolex Daytona | 8 | 6 | 8 |
| Rolex Submariner | 8 | 7 | 8 |
| Zenith El Primero | 8 | 7 | 8 |
| Overall | 40 | 34 (85%) | 40 (100%) |

The 92% target was exceeded on top-three retrieval for this five-family pilot and missed for top-one. With only five possible families, returning three gives a 60% uniform-random top-three baseline. Results cannot establish exact reference identification, generalization to unseen families, or performance on user phone photos. Every photo receives a ranked shortlist, including photos of watches outside the gallery. Earlier brand-and-box and price results are separate evaluations.

Reproduce from the project folder with python test_watch_families.py. The evaluation writes artifacts/family_test_results.csv. data/family_manifest.csv records original source paths, split and hashes. First run can download torchvision's pretrained ResNet-18 weights if they are absent. On Windows the OCR path uses Windows OCR.

# Overwatch
Local luxury-watch listing support system, based on Fractylz/Overwatch at commit d99e4cd.
## Start on Windows
Extract the ZIP into a writable folder. Install Python 3.12 or 3.13 (64-bit), then double-click RUN.cmd. The first launch installs dependencies and needs Internet access. The application opens at localhost. The trained detector and price model are already included.
If Python with the dependencies already exists, open PowerShell inside the folder and run:
```powershell
python -m streamlit run app.py --server.address localhost
```
## Try a listing
1. Upload examples/heldout_watch.jpg or your own photograph and select **Read watch photo**. The detector selects the highest-confidence watch in frame. Its box follows the dataset annotation, which can include the case and strap. Windows OCR reads the crop. Other platforms use EasyOCR and require its language weights.
2. Review the suggested brand and dial text. Add a listing title or reference if the dial is difficult to read. Type **116610LN** to try a catalogue reference (Rolex Submariner Date). Exact reference or model-name matches are reported with their evidence. An ambiguous dial can require manual confirmation.
3. Confirm or enter the brand and model, then enter any known attributes. Pilot asking-price estimates are available for catalogue families represented in the price sample. A detector class alone cannot establish the exact watch reference.
4. Save the listing. Saved listings persist in data/listings.sqlite on this computer. Export from the Saved listings tab. The saved record retains the photo filename, listing fields and estimate. Keep the original photograph separately.
## Detector
The trained best checkpoint is **artifacts/watch_detector.pt**. The source exports were Classify_Watch version 4 and watch-nefxk version 1. Both label 10 brands. Polygon labels were enclosed in axis-aligned boxes and brand names were aligned. For identical file hashes or repeated Roboflow base filenames, the held-out splits took priority. The preparation does not check every possible near-duplicate with a changed filename.
The split is 1,509 training, 381 validation and 189 test images. Classify_Watch supplied training examples only. The test set measures performance on the watch-nefxk export. It does not measure OCR, precise model identification or price accuracy. See artifacts/test_metrics.json for aggregate and per-class metrics and artifacts/detector_training.csv for the validation history.
Run the complete inference path from PowerShell:
```powershell
python e2e_prediction.py examples/heldout_watch.jpg
```
Rebuild from extracted YOLO exports, starting with a fresh data/detection directory:
```powershell
python prepare_datasets.py merge "path/to/Classify_Watch" "path/to/watch"
python prepare_datasets.py train
```
The train command defaults to the shipped architecture and size. On a CUDA-enabled PyTorch installation, use --device 0. For transfer learning you can supply --initial yolov8n.pt. New training replaces the local checkpoint. To evaluate the new run, install its checkpoint and update the report, run python evaluate_detector.py runs/brands (use the actual run folder). The app confidence threshold is selected on validation photos before counting test-set top-box decisions.
## Asking-price model
The supplemental price pilot uses 57 USD asking-price records (30 Rolex and 27 Omega). A public listing snapshot supplies brand, family, reference, year, condition, case material, diameter and box/papers. The three image datasets are independent of price supervision. The estimate is in USD. Thin model-family coverage, unknown attributes and the small pilot can lead to large errors.
The evaluation groups by brand and reference, with 43 training rows and 14 held-out rows. The final shipped XGBoost model was refit on all 57. Running the following retrains it from the packaged CSV:
```powershell
python price_classifier/train_price_model.py
```
## Image clustering
The application includes an offline clustering run on a local 2,453-photo archive. It uses ImageNet-pretrained ResNet-18 image embeddings, PCA and mini-batch K-means. Three clusters were selected from K=2 through 12 by sampled silhouette score (0.110). The file archive has 16 brand folders; its identity as the linked Kaggle release was not independently verified. Clusters describe visual similarity. Their numeric identifiers are unrelated to brand class IDs.
The upstream CLIP workflow is supplied as an editable notebook (outputs removed; credentials removed) and cluster_images.py. To run fresh CLIP embeddings on a photo folder:
```powershell
python -m pip install -r requirements-training.txt
python cluster_images.py "path/to/photos"
```
The CLIP path needs pretrained encoder weights on first run. The shipped run used cached ResNet-18 weights. To repeat it, run python cluster_archive.py "path/to/archive.zip".
## Sources
- Repository and clustering notebook: https://github.com/Fractylz/Overwatch/tree/d99e4cd00afdf59a63e572d099a9ec5916528307
- Classify_Watch by Special problem, version 4, CC BY 4.0: https://universe.roboflow.com/special-problem-ehhbd/classify_watch/dataset/4
- watch by project-jisou, version 1, CC BY 4.0. The held-out example photo is from this export: https://universe.roboflow.com/project-jisou/watch-nefxk/dataset/1
- Linked image dataset by Mathew Kouch: https://www.kaggle.com/datasets/mathewkouch/a-dataset-of-watches
- Additional USD listing data by malekoweis, chrono24-luxury-watch-listings subset: https://huggingface.co/datasets/malekoweis/resale-market-data

## Identify an unknown watch from a photo
Upload a front-facing watch photo and select **Read watch photo**. Compare the three possible families and select one if it matches. The family fills in both fields. Search covers Rolex Submariner, Rolex Daytona, Hublot Big Bang, Jaeger-LeCoultre Reverso and Zenith El Primero. Open Model evaluation for the photo-only results. Start with any of the sample photos under data/family_photos/test. See FAMILY_EVALUATION.md for the protocol and limitations. On first use the comparison feature may need Internet access to download torchvision ResNet-18 weights.

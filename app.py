import hashlib
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import pandas as pd
import streamlit as st
from listing import ROOT, catalogue as load_catalogue, bundle, estimate, identify, save_listing, history, visible_brands
from e2e_prediction import inspect_photo
from family_search import suggest

st.set_page_config(page_title="Overwatch | Watch listing studio", page_icon="⌚", layout="wide")

st.html("""
<style>
.stApp {
    --ow-bg: #101215;
    --ow-panel: #191d22;
    --ow-line: #30363e;
    --ow-text: #f4f1e9;
    --ow-muted: #bcc3cc;
    --ow-accent: #e3c489;
    background: var(--ow-bg);
    color: var(--ow-text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1320px; padding-top: 3.25rem; }
.stApp h1, .stApp h2, .stApp h3, .stApp label { color: var(--ow-text); }
.stApp [data-testid="stCaptionContainer"] p { color: var(--ow-muted); }
.ow-header {
    display: flex; align-items: center; gap: 14px;
    border-bottom: 1px solid var(--ow-line); padding-bottom: 20px;
}
.ow-mark {
    width: 36px; height: 36px; display: grid; place-items: center;
    border: 1px solid #806a47; border-radius: 50%;
    font: 600 17px Georgia, serif; color: var(--ow-accent);
}
.ow-brand { color: var(--ow-text); font-weight: 700; font-size: 17px; letter-spacing: .04em; }
.ow-hero { padding: 30px 0 12px; }
.ow-hero h1 { font: 500 clamp(32px, 5vw, 56px)/1.07 Georgia, serif; margin: 0 0 10px; }
.ow-hero p { color: var(--ow-muted); margin: 0; max-width: 62ch; }
.stApp [role="tablist"] { gap: 12px; border-bottom: 1px solid var(--ow-line); }
.stApp [role="tab"] { color: var(--ow-muted); padding: 0 12px; }
.stApp [role="tab"][aria-selected="true"] { color: var(--ow-accent); }
div[class*="st-key-photo-panel"], div[class*="st-key-details-panel"] {
    background: var(--ow-panel); border-radius: 14px;
}
.stApp [data-testid="stImage"] img { border-radius: 10px; }
.stApp [data-testid="stFileUploaderDropzone"] { border: 1px dashed #53606d; border-radius: 10px; }
.stApp button[kind="primary"] { background: var(--ow-accent); color: #20180b; border-color: var(--ow-accent); }
.stApp button[kind="primary"] p { color: #20180b; font-weight: 700; }
.stApp button:focus-visible { outline: 2px solid var(--ow-accent); outline-offset: 2px; }
@media(max-width: 650px) {
    [data-testid="stMainBlockContainer"] { padding-top: 2.6rem; }
    .stApp [role="tablist"] { gap: 2px; }
    .stApp [role="tab"] { padding: 0 8px; }
    .ow-hero { padding-top: 22px; }
}
</style>
<header class="ow-header"><span class="ow-mark" aria-hidden="true">O</span><span class="ow-brand">Overwatch</span></header>
<section class="ow-hero">
    <h1>Watch listing studio</h1>
    <p>Identify a watch, review comparable asking prices, and save a listing draft.</p>
</section>
""")
data = bundle()
catalogue = load_catalogue()
editor, saved, evaluation, groups = st.tabs(
    ["Create listing", "Saved listings", "Model evaluation", "Image clusters"]
)
with editor:
    left, right = st.columns([1, 1.22], gap="medium")
    with left, st.container(border=True, key="photo-panel"):
        st.subheader("Watch photo")
        photo = st.file_uploader("Watch photograph", type=["jpg", "jpeg", "png", "webp"])
        image = None
        photo_id = ""
        if photo:
            photo_id = hashlib.sha256(photo.getvalue()).hexdigest()
            try:
                with Image.open(BytesIO(photo.getvalue())) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                st.error("Choose a readable watch photograph.")
        if image and st.button("Read watch photo", type="primary"):
            with st.spinner("Reading dial text…"):
                scan_result = inspect_photo(image)
                scan_result["suggestions"] = suggest(image, " ".join(scan_result["ocr_text"]))
                st.session_state["analysis"] = (photo_id, scan_result)
        last_id, scan = st.session_state.get("analysis", (None, {}))
        scan = scan if last_id == photo_id else {}
        if image:
            preview = image.copy()
            if scan.get("box"):
                ImageDraw.Draw(preview).rectangle(scan["box"], outline="#02aa78", width=max(2, image.width//160))
            st.image(preview, width="stretch")
        for message in scan.get("messages", []):
            st.info(message)
        if scan.get("brand"):
            st.write(f"Detector: {scan['brand']} ({scan['brand_confidence']:.0%})")
        if scan.get("ocr_text"):
            st.write("Dial text", " · ".join(scan["ocr_text"]))
    with right, st.container(border=True, key="details-panel"):
        st.subheader("Listing details")
        suggestions = scan.get("suggestions", [])
        picked = None
        if suggestions:
            st.subheader("Possible watch families")
            st.caption("Compare the dial and case with the sample photos. Search covers five families.")
            from family_search import ROOT as photo_root
            for result, column in zip(suggestions, st.columns(len(suggestions))):
                with column:
                    st.image(str(photo_root / result["photo"]), width="stretch")
                    st.write(result["family"])
                    st.caption(result["evidence"])
            st.caption("If none match, try a sharper front photo with the dial centered.")
            picked = st.selectbox("Choose a family if it matches", [r["family"] for r in suggestions],
                                  index=None, key="choice"+photo_id)
        else:
            st.caption("Upload a photo for suggestions or enter the watch details below.")
        words = st.text_input("Listing title, dial text or reference number", key="text"+photo_id,
                              placeholder="e.g. Rolex 116610LN")
        combined = " ".join([*scan.get("ocr_text", []), words])
        candidate = identify(words, catalogue, scan.get("brand", "")) or identify(combined, catalogue, scan.get("brand", ""))
        if candidate:
            st.success(f"Catalogue match · {candidate['brand']} {candidate['model_family']} · {candidate['evidence']}")
        elif combined:
            st.info("No unique catalogue match. Confirm the brand and model below.")
        picked_brand, _, picked_model = picked.partition(" ") if picked else ("", "", "")
        if picked:
            candidate = dict(brand=picked_brand, model_family=picked_model, reference="", evidence="photo comparison")
        brands = sorted({r["brand"] for r in catalogue} | ({scan["brand"]} if scan.get("brand") else set()) | ({picked_brand} if picked else set()))
        named = visible_brands(combined, catalogue)
        confirmed_text_brand = next(iter(named)) if len(named) == 1 else None
        if confirmed_text_brand and scan.get("brand") and scan["brand"] != confirmed_text_brand:
            st.info(f"Text names {confirmed_text_brand}; the photo detector suggested {scan['brand']}. Confirm the brand below.")
        proposed = candidate["brand"] if candidate else (confirmed_text_brand or scan.get("brand"))
        brand = st.selectbox("Brand", brands, index=brands.index(proposed) if proposed in brands else None,
                             key=f"brand-{photo_id}-{proposed}", accept_new_options=True)
        families = sorted({r["model_family"] for r in catalogue if r["brand"] == brand})
        if picked and picked_brand == brand and picked_model not in families:
            families.append(picked_model)
        proposed_family = candidate["model_family"] if candidate and candidate["brand"] == brand else None
        family = st.selectbox("Model or family", families, index=families.index(proposed_family) if proposed_family in families else None,
                              key=f"family-{photo_id}-{brand}-{proposed_family}", accept_new_options=True)
        ref = st.text_input("Reference number (optional)", value=candidate["reference"] if candidate and candidate["model_family"] == family else "",
                            key=f"ref-{photo_id}-{family}-{candidate}")
        with st.expander("Listing attributes"):
            year_unknown = st.checkbox("Year unknown", value=True, key="year-unknown-"+photo_id)
            year = None
            if not year_unknown:
                year = st.number_input("Production year", 1750, 2100, value=None, key="year"+photo_id)
            size = st.number_input("Case diameter (mm)", 5., 100., value=None, key="size"+photo_id)
            condition = st.selectbox("Condition", ["Unknown","Used","New"], key="condition"+photo_id)
            material = st.selectbox("Case material", sorted({r["material"] for r in data["catalogue"]} | {"Unknown"}),
                                    index=None, key="material"+photo_id)
            papers = st.selectbox("Includes both box and papers", ["Unknown","Yes","No"], key="papers"+photo_id)
        if brand and family:
            st.subheader("Pricing evidence")
            pricing_rows = [row for row in data["catalogue"] if row["brand"] == brand]
            comparable_models = sorted({
                row["model_family"] for row in pricing_rows
                if row["model_family"] != family
                and row["model_family"].casefold().startswith(family.casefold() + " ")
            })
            variant = None
            if comparable_models:
                variant = st.selectbox(
                    "Confirm a variant (optional)", comparable_models, index=None,
                    placeholder="Variant unknown", key=f"variant-{photo_id}-{brand}-{family}",
                )
            pricing_family = variant or family
            rows = [row for row in pricing_rows if row["model_family"] == pricing_family]
            possible = not rows and bool(comparable_models)
            if possible:
                rows = [row for row in pricing_rows if row["model_family"] in comparable_models]

            record = dict(brand=brand, model_family=pricing_family, reference=ref, year=year, size_mm=size,
                          condition=condition, material=material, box_papers=papers)
            price = estimate(record)
            if price:
                with st.container(border=True):
                    st.metric("Pilot asking-price estimate", f"{price['currency']} {price['amount']:,.0f}")
                    st.caption(
                        f"{price['examples']} comparable listings · "
                        f"{price['currency']} {price['observed_min']:,.0f}–{price['observed_max']:,.0f} observed"
                    )
                if year is None:
                    st.caption("Estimate calculated with the watch year unknown.")
            elif possible:
                st.info("Review these possible comparables or confirm a variant to get a pilot estimate.")
            else:
                st.info("There are no pricing examples for this selection. You can still save a draft.")

            if rows:
                st.write("Possible comparables" if possible else "Comparable listings")
                st.caption("Historical asking prices and reported watch years from the dataset.")
                st.dataframe([
                    {"Model": row["model_family"], "Reference": row["reference"],
                     "Reported year": "Unknown" if pd.isna(row["year"]) else str(int(row["year"])),
                     "Asking price": f"{row['currency']} {row['price']:,.0f}"}
                    for row in sorted(rows, key=lambda row: row["price"])
                ], hide_index=True)

            if st.button("Save draft", type="primary", width="stretch"):
                listing_id = save_listing(dict(**record, estimated_price=price, photo_name=photo.name if photo else None,
                                              catalogue_evidence=candidate if candidate and candidate["brand"]==brand and candidate["model_family"]==pricing_family else None))
                st.success(f"Saved draft #{listing_id}")
with saved:
    st.subheader("Listing drafts")
    records = history()
    if records:
        frame = pd.json_normalize(records)
        frame["Watch"] = [
            " ".join(str(record.get(field) or "") for field in ("brand", "model_family")).strip()
            for record in records
        ]
        query = st.text_input("Search drafts", placeholder="Search by brand, model, or reference")
        if query.strip():
            references = frame.get("reference", pd.Series("", index=frame.index)).fillna("").astype(str)
            matches = (frame["Watch"] + " " + references).str.contains(query.strip(), case=False, regex=False)
            frame = frame[matches]
        fields = {
            "id": "Draft", "Watch": "Watch", "reference": "Reference", "year": "Year",
            "estimated_price.amount": "Asking-price estimate", "estimated_price.currency": "Currency",
            "saved": "Saved",
        }
        if frame.empty:
            st.info("No drafts match your search.")
        else:
            st.dataframe(frame.reindex(columns=list(fields)).rename(columns=fields), hide_index=True)
        st.download_button("Download listing history", json.dumps(records, indent=2), "watch-listings.json")
    else:
        st.write("Saved listings will appear here.")
with evaluation:
    r = data["report"]
    st.subheader("Photo-only family identification")
    test_path = ROOT / "artifacts/family_test_results.csv"
    if test_path.exists():
        family_test = pd.read_csv(test_path)
        st.write(f"Top-1 {family_test.top1.mean():.1%} · Top-3 {family_test.top3.mean():.1%} · {len(family_test)} held-out photographs")
        st.caption("Five supported watch families. Scores measure family ranking on the evaluation photos.")
        st.download_button("Download photo test results", test_path.read_bytes(), "family_test_results.csv")
    st.subheader("Asking-price pilot")
    st.write(f"{r['rows']} USD listings. Evaluation held out {r['holdout_rows']} listings across {r['held_out_references']} reference numbers. The shipped model was then refitted on all rows.")
    st.dataframe(pd.DataFrame([{"Metric":"Mean absolute error", "USD":r["mae"]},
                               {"Metric":"Root mean squared error", "USD":r["rmse"]},
                               {"Metric":"Training-median baseline MAE", "USD":r["median_baseline_mae"]}]), hide_index=True)
    st.caption("Dataset sources include resale-market-data and, when installed, the WatchFinder listing export.")
    st.subheader("Watch brand detector")
    summary_path = ROOT / "artifacts/test_metrics.json"
    if summary_path.exists():
        stats = json.loads(summary_path.read_text())
        st.write(f"Independent test split · {stats['test_images']} images · {stats['classes']} brand labels")
        hit = stats.get("app_top_box")
        if hit:
            st.metric("Correct brand and box on test photos", f"{hit['correct']} / {hit['total']}")
            st.caption(f"Top suggestion at confidence {hit['confidence_threshold']:.1%}. Required overlap with the labelled watch: IoU >= 0.50. No suggestion on {hit['no_detection']} photos.")
        st.dataframe(pd.DataFrame([stats["overall"]]), hide_index=True)
        st.dataframe(pd.DataFrame(stats["per_brand"]), hide_index=True)
        metrics = pd.read_csv(ROOT / "artifacts/detector_training.csv")
        st.line_chart(metrics.set_index("epoch")[["metrics/mAP50(B)", "metrics/mAP50-95(B)"]])
        st.caption("The chart shows validation results for each training epoch. The tables show test results for the selected checkpoint.")
    else:
        st.write("Detector evaluation has not been saved yet.")
with groups:
    group_file = ROOT / "artifacts/visual_groups.json"
    if group_file.exists():
        summary = json.loads(group_file.read_text())
        st.subheader("Groups of similar product photographs")
        st.write(f"{summary['images']:,} photos · {summary['groups']} groups · silhouette {summary['silhouette']:.3f}")
        st.caption(summary["method"])
        st.image(str(ROOT / "artifacts/visual_groups.jpg"), width=400)
        st.download_button("Download group assignments", (ROOT / "artifacts/visual_groups.csv").read_bytes(), "visual_groups.csv")
    else:
        st.write("Run cluster_archive.py on a ZIP of product photos to view the groups here.")

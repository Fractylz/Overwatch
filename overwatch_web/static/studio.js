const byId = id => document.getElementById(id);
const listing = byId("listing");
const photo = byId("photo");
let models = {};
let drafts = [];
let previewUrl;
const field = key => listing.elements.namedItem(key);
const status = (id, value) => { byId(id).textContent = value; };
const money = (amount, currency) => currency + " " + Number(amount).toLocaleString(undefined, {maximumFractionDigits: 0});

function element(tag, value) {
    const node = document.createElement(tag);
    node.textContent = value ?? "";
    return node;
}

async function api(url, body) {
    const options = body instanceof FormData ? {method: "POST", body} : body ? {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
    } : {};
    const response = await fetch(url, options);
    if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.error || (response.status === 413 ? "Choose a photo under 12 MB." : "Could not complete the request. Check the fields and try again."));
    }
    return response.json();
}

function options(id, labels) {
    byId(id).replaceChildren(...labels.map(label => {
        const option = element("option", "");
        option.value = label;
        return option;
    }));
}

function applyMatch(match) {
    field("brand").value = match.brand;
    options("families", models[match.brand] || []);
    field("model_family").value = match.model_family;
    field("reference").value = match.reference || "";
    byId("pricing").hidden = true;
}

function payload() {
    return {...Object.fromEntries(new FormData(listing)), photo_name: photo.files[0]?.name};
}

async function perform(button, statusId, pending, action) {
    button.disabled = true;
    status(statusId, pending);
    try { await action(); } catch (error) { status(statusId, error.message); }
    finally { button.disabled = false; }
}

photo.addEventListener("change", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = photo.files[0];
    byId("preview").hidden = !file;
    if (file) {
        previewUrl = URL.createObjectURL(file);
        byId("preview").querySelector("img").src = previewUrl;
    }
    byId("box").hidden = true;
    byId("suggestions").replaceChildren();
    status("scan-status", "");
});

byId("scan-form").addEventListener("submit", event => {
    event.preventDefault();
    const file = photo.files[0];
    if (!file) return;
    perform(event.submitter, "scan-status", "Analyzing the watch…", async () => {
        const data = new FormData();
        data.append("photo", file);
        const scan = await api("/api/scan", data);
        if (file !== photo.files[0]) return;
        if (scan.box) {
            const [x0, y0, x1, y1] = scan.box;
            const box = byId("box");
            box.style.left = x0 / scan.image_width * 100 + "%";
            box.style.top = y0 / scan.image_height * 100 + "%";
            box.style.width = (x1 - x0) / scan.image_width * 100 + "%";
            box.style.height = (y1 - y0) / scan.image_height * 100 + "%";
            box.hidden = false;
        }
        const messages = [scan.brand && scan.brand + " · " + Math.round(scan.brand_confidence * 100) + "% detector confidence"];
        if (scan.ocr_text.length) messages.push("Dial text: " + scan.ocr_text.join(" · "));
        status("scan-status", [...messages, ...scan.messages].filter(Boolean).join(" — ") || "Confirm your watch details.");
        if (scan.match) applyMatch(scan.match);
        else if (scan.brand) {
            field("brand").value = scan.brand;
            field("model_family").value = "";
            field("reference").value = "";
            options("families", models[scan.brand] || []);
            byId("pricing").hidden = true;
        }
        byId("suggestions").replaceChildren(...scan.suggestions.map(hit => {
            const button = element("button", hit.family);
            button.className = "secondary";
            const image = element("img", "");
            image.alt = "";
            image.loading = "lazy";
            image.src = "/gallery/" + encodeURIComponent(hit.photo.split("/").at(-1));
            button.prepend(image);
            button.onclick = () => {
                const brand = Object.keys(models).sort((a, b) => b.length - a.length).find(name => hit.family.startsWith(name + " "));
                if (brand) applyMatch({brand, model_family: hit.family.slice(brand.length + 1)});
            };
            return button;
        }));
    });
});

byId("match-form").addEventListener("submit", event => {
    event.preventDefault();
    perform(event.submitter, "match-status", "Searching the catalogue…", async () => {
        const match = await api("/api/match", {text: byId("lookup").value, brand: field("brand").value});
        if (match) applyMatch(match);
        status("match-status", match ? "Catalogue match · " + match.brand + " " + match.model_family : "No unique match. Enter the watch details below.");
    });
});

function table(headers, rows) {
    const wrap = element("div", "");
    wrap.className = "table-wrap";
    const grid = wrap.appendChild(element("table", ""));
    const head = grid.createTHead().insertRow();
    headers.forEach(header => head.appendChild(element("th", header)).scope = "col");
    const body = grid.createTBody();
    rows.forEach(values => {
        const row = body.insertRow();
        values.forEach(value => { row.insertCell().textContent = value ?? "Unknown"; });
    });
    return wrap;
}

listing.addEventListener("input", () => {
    byId("pricing").hidden = true;
    options("families", models[field("brand").value] || []);
});

async function review(info) {
    const result = await api("/api/price", info);
    if (JSON.stringify(info) !== JSON.stringify(payload())) return;
    const panel = byId("pricing");
    panel.replaceChildren(element("h2", "Pricing evidence"));
    const price = result.estimate;
    if (price) {
        panel.append(element("p", "Pilot asking-price estimate"));
        const amount = element("p", money(price.amount, price.currency));
        amount.className = "amount";
        panel.append(amount, element("p", price.examples + " comparable listings · " +
            money(price.observed_min, price.currency) + "–" + money(price.observed_max, price.currency) + " observed"));
    } else {
        panel.append(element("p", result.comparables.length ? "Possible comparables for related variants." : "No pricing examples for this selection. You can still save a draft."));
    }
    result.related.forEach(family => {
        const button = element("button", "Confirm " + family);
        button.className = "secondary";
        button.onclick = () => { field("model_family").value = family; listing.requestSubmit(); };
        panel.append(button);
    });
    if (result.comparables.length) panel.append(table(
        ["Model", "Reference", "Reported year", "Asking price"],
        result.comparables.map(row => [row.family, row.reference, row.year, money(row.price, row.currency)]),
    ));
    panel.hidden = false;
}

listing.addEventListener("submit", event => {
    event.preventDefault();
    perform(event.submitter || listing.querySelector("button[type=submit]"), "listing-status", "Checking comparable listings…", async () => {
        await review(payload());
        status("listing-status", "");
    });
});

function drawDrafts() {
    const query = byId("draft-search").value.trim().toLowerCase();
    const rows = drafts.filter(row => [row.brand, row.model_family, row.reference].join(" ").toLowerCase().includes(query));
    byId("draft-table").replaceChildren(rows.length ? table(
        ["Draft", "Watch", "Reference", "Year", "Estimate", "Saved"],
        rows.map(row => [row.id, [row.brand, row.model_family].filter(Boolean).join(" "), row.reference || "Unknown", row.year,
            row.estimated_price ? money(row.estimated_price.amount, row.estimated_price.currency) : "Unavailable", row.saved]),
    ) : element("p", query ? "No drafts match your search." : "Saved drafts will appear here."));
}

async function refreshDrafts() {
    drafts = await api("/api/drafts");
    drawDrafts();
    const link = byId("export");
    if (link.href.startsWith("blob:")) URL.revokeObjectURL(link.href);
    link.href = URL.createObjectURL(new Blob([JSON.stringify(drafts, null, 2)], {type: "application/json"}));
    link.hidden = !drafts.length;
}

byId("draft-search").addEventListener("input", drawDrafts);
byId("save").onclick = event => {
    if (!listing.reportValidity()) return;
    perform(event.currentTarget, "listing-status", "Saving draft…", async () => {
        const saved = await api("/api/drafts", payload());
        status("listing-status", "Saved draft #" + saved.id);
        await refreshDrafts().catch(() => status("listing-status", "Saved draft #" + saved.id + ". Reload the page to refresh saved drafts."));
    });
};

api("/api/catalogue").then(data => {
    models = data;
    options("brands", Object.keys(data).sort());
}).catch(error => status("match-status", error.message));
refreshDrafts().catch(error => byId("draft-table").replaceChildren(element("p", error.message)));
api("/api/evaluation").then(data => {
    const hit = data.detector?.app_top_box;
    const rows = [
        hit && ["Brand and box", hit.correct + "/" + hit.total + " test photos"],
        data.family && ["Photo-only family ranking", "Top 1: " + data.family.top1 + "/" + data.family.photos + " · Top 3: " + data.family.top3 + "/" + data.family.photos],
        data.pricing && ["Asking-price evaluation", money(data.pricing.mae, data.pricing.currency) + " mean absolute error · " + data.pricing.holdout_rows + " held-out listings"],
        data.clusters && ["Photo clusters", data.clusters.images + " photos · " + data.clusters.groups + " groups"],
    ];
    byId("metrics").replaceChildren(...rows.filter(Boolean).map(([title, value]) => {
        const section = element("section", "");
        section.append(element("h3", title), element("p", value));
        return section;
    }));
}).catch(error => byId("metrics").replaceChildren(element("p", error.message)));

import base64
import csv
import itertools
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from enhance import (
    COLOR_SPACES,
    CONTRAST_METHODS,
    METRIC_INFO,
    SR_METHODS,
    UPSAMPLE_METHODS,
    available_sr_methods,
    build_comparison_image,
    compute_metrics,
    enhance,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

OUTPUTS = "outputs"
os.makedirs(UPLOADS := "uploads", exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

# A full cross-product of every option is 7 x 12 x 8 x 8 = 5376 runs, and the
# two SRGAN paths alone cost seconds each on CPU. The sweep therefore caps how
# many combinations it will run and tells the UI when it truncated, instead of
# silently starting a job nobody can wait out.
MAX_SWEEP_RUNS = 64


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def _thumb_b64(img_bgr: np.ndarray, size: int = 190) -> str:
    """Small JPEG thumbnail — sweep tables carry dozens of these."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def _read_image(file_storage, work_size: int):
    """Decode an upload and resize to work_size x work_size (paper: 512x512)."""
    buf = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (work_size, work_size))


def _common_params(form) -> dict:
    """Pipeline parameters shared by /enhance and /compare."""
    tile = int(float(form.get("tile_size", 8)))
    return {
        "gamma":                   float(form.get("gamma", 0.8)),
        "clip_limit":              float(form.get("clip_limit", 2.0)),
        "tile_size":               (tile, tile),
        "cn":                      float(form.get("cn", 0.85)),
        "sigma":                   float(form.get("sigma", 1.0)),
        "agcwd_alpha":             float(form.get("agcwd_alpha", 0.5)),
        "tophat_ksize":            int(float(form.get("tophat_ksize", 15))),
        "use_color_transfer":      form.get("use_color_transfer", "true").lower() == "true",
        "color_transfer_strength": float(form.get("color_transfer_strength", 1.0)),
        "use_visual_punch":        form.get("use_visual_punch", "false").lower() == "true",
        "visual_punch_saturation": float(form.get("visual_punch_saturation", 1.6)),
        "visual_punch_warmth":     float(form.get("visual_punch_warmth", 22.0)),
    }


def _flatten_metrics(metrics: dict) -> dict:
    """metrics dict -> flat {key: value} for CSV rows and table cells."""
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for block in ("original", "enhanced", "gain"):
        for k, v in metrics.get(block, {}).items():
            flat[f"{block}_{k}" if block != "gain" else k] = v
    return flat


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/options")
def options():
    """Everything the UI needs to build its selectors, straight from enhance.py."""
    avail = available_sr_methods()
    return jsonify({
        "color_spaces": [{"id": k, "label": v[0], "desc": v[1]}
                         for k, v in COLOR_SPACES.items()],
        "contrast_methods": [{"id": k, "label": v[0], "desc": v[1]}
                             for k, v in CONTRAST_METHODS.items()],
        "upsample_methods": [{"id": k, "label": v[0], "desc": v[1]}
                             for k, v in UPSAMPLE_METHODS.items()],
        "sr_methods": [{"id": k, "label": v[0], "desc": v[1],
                        "available": avail.get(k, True)}
                       for k, v in SR_METHODS.items()],
        "metric_info": METRIC_INFO,
        "max_sweep_runs": MAX_SWEEP_RUNS,
        "defaults": {"color_space": "hsv", "contrast_method": "clahe",
                     "upsample_method": "hu_wbi", "sr_method": "auto"},
    })


@app.route("/enhance", methods=["POST"])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": "Invalid or unsupported file"}), 400

    params    = _common_params(request.form)
    use_srgan = request.form.get("use_srgan", "true").lower() == "true"
    work_size = int(float(request.form.get("work_size", 512)))

    stage = {
        "color_space":     request.form.get("color_space", "hsv"),
        "contrast_method": request.form.get("contrast_method", "clahe"),
        "upsample_method": request.form.get("upsample_method", "hu_wbi"),
        "sr_method":       request.form.get("sr_method", "auto"),
    }
    for key, table in (("color_space", COLOR_SPACES),
                       ("contrast_method", CONTRAST_METHODS),
                       ("upsample_method", UPSAMPLE_METHODS),
                       ("sr_method", SR_METHODS)):
        if stage[key] not in table:
            return jsonify({"error": f"Unknown {key}: {stage[key]}"}), 400

    img_resized = _read_image(file, work_size)
    if img_resized is None:
        return jsonify({"error": "Could not decode image"}), 400

    t0 = time.time()
    enhanced, config = enhance(img_resized, use_srgan=use_srgan,
                               return_config=True, **params, **stage)
    elapsed = time.time() - t0

    metrics = compute_metrics(img_resized, enhanced)
    comparison = build_comparison_image(img_resized, enhanced, metrics, config)

    uid = uuid.uuid4().hex
    cv2.imwrite(os.path.join(OUTPUTS, f"{uid}_enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(OUTPUTS, f"{uid}_comparison.png"), comparison)

    # single-run metrics CSV, so one run is as exportable as a sweep
    csv_path = os.path.join(OUTPUTS, f"{uid}_metrics.csv")
    flat = _flatten_metrics(metrics)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in {**config, **flat}.items():
            w.writerow([k, v])

    return jsonify({
        "original":      _to_b64(img_resized),
        "enhanced":      _to_b64(enhanced),
        "comparison":    _to_b64(comparison),
        "metrics":       metrics,
        "config":        config,
        "elapsed":       round(elapsed, 3),
        "dl_enhanced":   f"{uid}_enhanced.png",
        "dl_comparison": f"{uid}_comparison.png",
        "dl_metrics":    f"{uid}_metrics.csv",
    })


@app.route("/compare", methods=["POST"])
def compare_variants():
    """
    Run every selected combination of the four swappable stages on one image
    and return a metrics row per combination.

    The UI multi-selects each axis; the cross product is capped at
    MAX_SWEEP_RUNS so a careless "select all" cannot hang the server. ΔE is
    skipped here (it dominates runtime) — everything else is identical to the
    single-run metrics, and rows stay directly comparable because all of them
    share the same input image, work size and slider values.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": "Invalid or unsupported file"}), 400

    def _list(field, table, fallback):
        raw = [v for v in request.form.get(field, "").split(",") if v]
        vals = [v for v in raw if v in table]
        return vals or [fallback]

    spaces   = _list("color_spaces",     COLOR_SPACES,     "hsv")
    contrast = _list("contrast_methods", CONTRAST_METHODS, "clahe")
    upsample = _list("upsample_methods", UPSAMPLE_METHODS, "hu_wbi")
    srs      = _list("sr_methods",       SR_METHODS,       "none")

    combos = list(itertools.product(spaces, upsample, contrast, srs))
    truncated = len(combos) > MAX_SWEEP_RUNS
    combos = combos[:MAX_SWEEP_RUNS]

    params    = _common_params(request.form)
    work_size = int(float(request.form.get("work_size", 256)))

    img = _read_image(file, work_size)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    rows = []
    for space, up, con, sr in combos:
        t0 = time.time()
        out, config = enhance(img, color_space=space, upsample_method=up,
                              contrast_method=con, sr_method=sr,
                              use_srgan=(sr != "none"), return_config=True,
                              **params)
        metrics = compute_metrics(img, out, include_delta_e=False)
        rows.append({
            "config":  config,
            "label":   " · ".join([COLOR_SPACES[space][0], UPSAMPLE_METHODS[up][0],
                                   CONTRAST_METHODS[con][0], SR_METHODS[sr][0]]),
            "metrics": metrics,
            "flat":    _flatten_metrics(metrics),
            "thumb":   _thumb_b64(out),
            "elapsed": round(time.time() - t0, 3),
        })

    # one CSV holding the whole table
    uid = uuid.uuid4().hex
    csv_path = os.path.join(OUTPUTS, f"{uid}_sweep.csv")
    if rows:
        cfg_cols  = ["color_space", "upsample_method", "contrast_method",
                     "sr_method", "sr_actual"]
        met_cols  = list(rows[0]["flat"].keys())
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cfg_cols + ["seconds"] + met_cols)
            for r in rows:
                w.writerow([r["config"].get(c) for c in cfg_cols]
                           + [r["elapsed"]]
                           + [r["flat"].get(c) for c in met_cols])

    return jsonify({
        "original":   _to_b64(img),
        "rows":       rows,
        "truncated":  truncated,
        "count":      len(rows),
        "work_size":  work_size,
        "dl_csv":     f"{uid}_sweep.csv",
    })


@app.route("/download/<filename>")
def download(filename):
    # Uploads never touch this path, but the filename still comes from the
    # client, so keep it inside OUTPUTS.
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUTS, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    if safe.endswith(".csv"):
        return send_file(path, mimetype="text/csv",
                         download_name=safe, as_attachment=True)
    return send_file(path, mimetype="image/png",
                     download_name=safe, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

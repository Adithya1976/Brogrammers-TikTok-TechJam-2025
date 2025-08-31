
# VeilVault — Privacy Meets AI (Images & Video)

[video](https://www.youtube.com/watch?v=xHC2i59YnKc)

> A fast, practical pipeline for **detecting**, **masking**, and **de-identifying** sensitive content in images and videos. We blend **classical key-frame sampling** with **modern vision + NLP** so you get near real-time performance and pixel-accurate control — tap to blur or unblur any entity, anytime.

* **Stack:** FastAPI, Python, OpenCV, GroundingDINO + SAM (Grounded-SAM), Tesseract/EasyOCR, Microsoft Presidio, optional Geo-adversarial noise inspired by GeoCLIP. ([GitHub][1])
* **Run:** `pip install -r requirements.txt` → `python main.py`. A minimal client is in `test_api.py`. ([GitHub][1])

---

## Why this exists

Most “privacy filters” are all-or-nothing. We do it differently: for **every detected entity** we return a **binary mask** alongside the **entity name**, so your app can **toggle blur/unblur** with one tap. This gives users **full autonomy** while keeping server compute low via **key-frame sampling** on video.

---

## The Story — Three Stages

### 1) See the scene (GroundingDINO + SAM)

**Imagine a street photo** — faces, license plates, storefront signs, paper documents, screens. Before reading any text, we **ask the image what’s there**:

* **GroundingDINO** proposes regions using open-vocabulary prompts (e.g., “face”, “license plate”, “screen”, “ID card”, “document”, “signboard”).
* **SAM** (Segment Anything) snaps those proposals into **precise masks**.

This stage gives us **object-level masks** that work even when text is faint, angled, or partially occluded — a strong prior before OCR.

> Why: it reduces OCR overreach (scratches ≠ letters), anchors PII to **real object regions**, and supplies masks that are already click-ready.

*(Repo tech list references Grounded-SAM components.)* ([GitHub][1])

---

### 2) Read & reason (OCR + Presidio PII)

Next, we **read** what matters:

* **OCR** (Tesseract/EasyOCR depending on config) extracts words/lines + their boxes.
* **Presidio Analyzer** classifies spans into PII types — `EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `CREDIT_CARD`, `SSN`, `LICENSE_PLATE`, etc.
* We **align tokens ↔ spans** and **clip** them to the best supporting region masks from Stage 1 (when available).
* Output is **strictly filtered**: each entity entry contains only:

  * `entity_name` (e.g., `EMAIL_ADDRESS`)
  * `mask` (binary, same H×W as input; encoded for transport)

> Why: keeping the payload to **name + mask** is intentional — **no raw text leaves the server** by default; clients still get pixel-exact control.

*(Repo already lists Tesseract + Presidio among core tech.)* ([GitHub][1])

---

### 3) Hide the world (Geo-adversarial noise)

Some images leak **location** (a skyline, a unique mural, even the texture of a bus stop). We apply targeted, low-visibility **adversarial noise** designed to **confuse SOTA geolocation models** (e.g., GeoCLIP-style). Masks from Stage 1 ensure we **only perturb** location-telling **background regions** — not faces or text the user cares to keep readable.

> Result: **Humans notice nothing**, but geolocation classifiers struggle. Opt-in per request.

*(Your README mentions GeoCLIP in the stack; this stage aligns with that.)* ([GitHub][1])

---

## Video: Hybrid key-frame sampling (classical + AI)

Video is where compute goes to die — unless you’re picky. We are:

1. **Classical sampling:**

   * **Uniform stride** (e.g., every *K* frames), and/or
   * **Motion/scene deltas** (histogram diffs) to skip near-duplicates.
2. **AI on key frames:** Run the full **3-stage pipeline** only on sampled frames.
3. **Temporal linking:** Shallow IoU association between consecutive processed frames to keep a stable `entity_name` label per stream of masks.

> You get **big speedups** and **tiny costs**, with masks on the frames that matter most — and therefore smooth, tap-to-blur UX even on long clips.

---

## What the API returns (lean & click-ready)

> **Design principle:** minimal, safe, useful.
> Each detected entity is **just** `{ entity_name, mask }`.

### Image response (example)

```json
{
  "type": "image",
  "width": 1920,
  "height": 1080,
  "entities": [
    {
      "entity_name": "EMAIL_ADDRESS",
      "mask": { "format": "png", "data_base64": "<1ch grayscale PNG, HxW>" }
    },
    {
      "entity_name": "LICENSE_PLATE",
      "mask": { "format": "rle", "counts": "<COCO RLE>", "size": [1080,1920] }
    }
  ]
}
```

### Video response (example)

```json
{
  "type": "video",
  "fps": 29.97,
  "frames": [
    {
      "index": 31,
      "t": 1.033,
      "entities": [
        { "entity_name": "PERSON", "mask": { "format": "png", "data_base64": "<...>" } },
        { "entity_name": "LICENSE_PLATE", "mask": { "format": "png", "data_base64": "<...>" } }
      ]
    }
  ]
}
```

> **Encoding:** We support **PNG (base64)** or **COCO-RLE** for masks to avoid shipping raw NumPy arrays.

---

## API — endpoints & usage

> Server setup in this repo: `pip install -r requirements.txt` → `python main.py`. ([GitHub][1])

### `POST /process`

Upload one file (`image/*` or `video/*`). Returns a job handle.

* **Form field:** `file=@/path/to/file.jpg|mp4`
* **Response:**

  ```json
  { "job_id": "abc123" }
  ```

**cURL**

```bash
curl -F "file=@/path/to/photo.jpg" http://127.0.0.1:8000/process
```

### `GET /status/{job_id}`

Lightweight status poll.

```json
{ "job_id": "abc123", "status": "queued|running|done|error" }
```

### `GET /result/{job_id}`

Final structured JSON (see examples above).

**cURL**

```bash
curl http://127.0.0.1:8000/result/abc123 > result.json
```

> Notes
> • For videos, results include only sampled frames.
> • To tune sampling, see `config.py` (stride, delta thresholds).
> • Stage 3 (geo-noise) is opt-in via request flag or server config.

---

## Client integration: one-tap blur/unblur

1. **Render** each `mask` as a transparent overlay.
2. **On tap**, hit-test the pixel → find intersecting entity → toggle that mask.
3. **Blur** (Gaussian/pixelate) where `mask==1`. **Unblur** by removing overlay.
4. **Video:** keep a `{frame_index -> entity masks}` map; for smoothness, hold last known mask across skipped frames or interpolate with simple dilation.

> Because we return **one mask per entity**, users decide **what** to hide and **when** — no hard-coded server policy.

---

## Config & knobs (quick)

* **Stage 1 (GroundingDINO + SAM):**

  * Prompt set (objects of interest), NMS thresholds, SAM point/box mode.
* **Stage 2 (OCR + Presidio):**

  * OCR `low_text`, `text_threshold`, min size; Presidio recognizers on/off.
* **Stage 3 (Geo-noise):**

  * Enable/disable; noise budget; region-of-interest (exclude faces/text).
* **Video:**

  * `stride K`, histogram-delta threshold, IoU threshold for temporal linking.

*(A subset of these are already hinted in your repo’s Tech/Setup docs.)* ([GitHub][1])

---

## Quick Start

```bash
# 1) Install
pip install -r requirements.txt

# 2) Run server
python main.py

# 3) Try it (image)
python test_api.py --file sample_private.jpg

# 4) Try it (video)
python test_api.py --file demo.mp4
```

*(Your README shows similar steps; keep using `test_api.py` as the template client.)* ([GitHub][1])

---

## Roadmap

* Lightweight **face/license-plate detectors** to complement GroundingDINO prompts.
* Optional **on-device** client-side blur for zero-trust deployments.
* **Track IDs** for entities across video frames (beyond names).
* Rich **admin dashboard** for metrics (latency, false-positive review).

---

## Security & Privacy

* We return only **`entity_name` + `mask`**; no raw OCR text is required in responses.
* Images/videos need not be stored server-side beyond processing.
* Geo-adversarial noise is **opt-in** and bypassed for high-fidelity exports.

---

## License

MIT (unless specified otherwise in this repository).

---

# Testing the fusion-svc API with Postman

A step-by-step guide for QA/testers. No coding needed. The API is the FastAPI
sidecar in [`fusion_svc/app.py`](../fusion_svc/app.py) (see
[`fusion-svc-API.md`](fusion-svc-API.md) for the developer reference). It has
**2 endpoints**: `GET /health`, `POST /fuse`.

`/fuse` merges a **burst of crops of one license plate** into a single restored
plate image (PNG). You upload N same-size crops; you get one fused plate back.

> **Direct vs via main API.** This guide tests the sidecar **directly** (port
> `8100`), which returns the fused plate **image**. The **main traffic API**
> (port `7862`) also reaches these engines through `POST /predict/plates/multiframe`
> and `POST /predict/plates/video`, but returns **OCR text (JSON)** instead of an
> image and auto-resizes the burst. Test those against `7862`, not `8100`.
>
> This sidecar is **optional** — the main API runs these engines in-process by
> default. Start it only if you specifically want to test fusion in isolation.

## 0. Start the API (standalone sidecar)

```bash
git submodule update --init --recursive      # fresh clone only
uv sync --directory fusion_svc
uv run --directory fusion_svc uvicorn fusion_svc.app:app --host 127.0.0.1 --port 8100
```

Confirm it's up: `GET http://127.0.0.1:8100/health` → `{"status":"ok"}`.

You can either **import the ready-made collection** (fastest) or build the
requests by hand. Both are covered.

---

## 1. Import the collection (recommended)

1. In Postman: **File → Import** (or the **Import** button, top-left).
2. Select **[`fusion-svc.postman_collection.json`](fusion-svc.postman_collection.json)**
   from this repo.
3. A collection **"fusion-svc API"** appears in the left sidebar with all
   requests pre-built.
4. Set the base URL once for the whole collection:
   - Click the collection name → **Variables** tab.
   - Set **`base_url`** Current Value to your server, e.g. `http://127.0.0.1:8100`
     (no trailing slash). **Save** (Ctrl/Cmd-S).
5. For any `fuse` request, you must re-attach your own files (Postman does not
   store file contents): open the request → **Body → form-data** → on each
   `files` row click **Select Files** and pick a plate crop.

Then jump to **§3 Test scenarios**.

---

## 2. Build requests by hand (if not importing)

Create a new collection, add a variable `base_url = http://127.0.0.1:8100`, then
add the requests below. URLs use `{{base_url}}`.

### 2a. Health check — `GET {{base_url}}/health`

No params. Send. Expect `200` and exactly:

```json
{ "status": "ok" }
```

**This is your first test — if it fails, nothing else will work.**

### 2b. Fuse — `POST {{base_url}}/fuse`

- Method **POST**, URL `{{base_url}}/fuse`.
- **Body** tab → select **form-data**.
- Add a row: **Key** = `files`, and on the right of the key field change the
  type dropdown from _Text_ to **File**. Then **Select Files** → pick a crop.
- Add **more `files` rows** (each File type) for the rest of the burst — you
  need **2+ same-size crops** of the *same* plate for fusion to do anything.
- **Params** tab → add `engine` = `mflpr2` (and optionally `scale` = `2`).
- Send. You get an **annotated-free restored plate PNG** back (see §4 to view/save).

> The field name **must be `files`** (plural), one row per crop, all set to File
> type. All crops must be the **same pixel size** and ordered.

---

## 3. Test scenarios

`/fuse` output depends on `engine` and `scale`. Use 2+ same-size crops of one
plate. Run these in order.

| #   | files            | Query params              | Expected response                                              |
| --- | ---------------- | ------------------------- | -------------------------------------------------------------- |
| 1   | 3 crops (96×32)  | `engine=mflpr2`           | `200`, **PNG**, color restored plate (same size as input)      |
| 2   | 3 crops (96×32)  | `engine=mflpr2&scale=2`   | `200`, **PNG** at **2× size** (192×64) — sharper               |
| 3   | 3 crops          | `engine=eott`             | `200`, **PNG**, **binarized** (black/white) plate, fixed grid  |
| 4   | 3 crops          | `engine=eott&scale=2`     | `200`, **PNG** — same size as scenario 3 (**eott ignores scale**) |
| 5   | 1 crop           | `engine=mflpr2`           | `200`, **PNG** (single frame, nothing to fuse — passthrough)   |
| 6   | 3 crops          | `engine=bogus`            | **`400`**, `unknown engine: 'bogus'`                           |
| 7   | a `.txt`/garbage | `engine=mflpr2`           | **`400`**, `no decodable frames`                               |
| 8   | (no files row)   | `engine=mflpr2`           | **`422`**, FastAPI validation error (missing `files`)          |
| 9   | crops of **different sizes** | `engine=mflpr2` | **`500`**, `mflpr2 failed: ...` (engine needs equal-size frames) |

> Scenarios 1 vs 3 are the key comparison: **mflpr2** returns a *color* restored
> plate that scales; **eott** returns a *binarized* plate on a fixed grid and
> ignores `scale`. Pick by what your downstream OCR expects.

---

## 4. Handling the binary (PNG) response

Every successful `/fuse` returns an image, not JSON. Postman shows it in the
response pane:

- The **Preview** tab renders the fused PNG.
- To save: click **⋯ / Save Response → Save to a file**, give it a `.png`
  extension.

Errors (`400`/`422`/`500`) come back as **JSON** in the response body — read the
`detail` field for the reason.

---

## 5. Per-request parameters (query)

Pass these in the **Params** tab. Both optional.

| Param    | Type | Values            | Meaning                                                       |
| -------- | ---- | ----------------- | ------------------------------------------------------------- |
| `engine` | string | `mflpr2` \| `eott` | fusion engine (default `mflpr2`). `eott` outputs binarized.  |
| `scale`  | int  | `1`–`8` (default `1`) | upscale factor — **applied by `mflpr2`, ignored by `eott`** |

There is no model/weights selection and no env config to worry about — the two
engines are fixed and always available once the server is up.

---

## 6. Bug-report checklist

When something looks wrong, capture for the dev:

- The **request**: full URL incl. `engine`/`scale`, and how many crops + their
  pixel size.
- The **response**: status code, and the JSON `detail` (for errors) or the saved
  PNG (for visual issues).
- `GET /health` output (confirms the server is up).
- Expected vs actual (e.g. "fused plate is blurry", "eott output all black",
  "scale had no effect").

Common, **not-a-bug** results:

- `400 unknown engine '...'` → typo in `engine`; use `mflpr2` or `eott`.
- `400 no decodable frames` → the uploaded file wasn't a valid image.
- `500 ... failed` with **different-size crops** → crop all frames to the same
  size first; the engines register equal-size frames.
- `eott` output ignores `scale` and looks **binarized** (black/white) → that is
  the native eott pre-OCR result, not a bug. Use `mflpr2` for a color plate.
- A **single** crop returns nearly the input → nothing to fuse from one frame.

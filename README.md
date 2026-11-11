# Leaf classifier — live camera demo (mlr-ex08)

**Live page: <https://vasutova.github.io/mlr-ex08/>**

Point your phone camera at a leaf and the four class probabilities update live.
Everything runs in the browser: the model is `leaf_model.onnx`, executed by
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM) loaded
from a CDN. No server, no image upload anywhere — the camera frames never leave
the device.

Scan the QR code at the bottom of the page to open it on a phone. The code is
drawn by `index.html` itself (a small QR encoder is included in the file), so no
third-party QR-image service is involved.

## What the page does

1. `getUserMedia({ video: { facingMode: { ideal: "environment" } } })` — rear camera.
2. Takes the **central square** of the current frame.
3. Resizes it to **96 × 96** (`input_size` from `model_info.json`).
4. Scales RGB samples to **0–1**.
5. Standardises each channel with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`.
6. Feeds a planar **CHW `float32` tensor of shape 1 × 3 × 96 × 96** to the input
   named `image`.
7. Applies softmax to the `logits` output and shows the probability of **every**
   class, highest first, updating each frame.

Large photos are reduced by halving on the way down to 96 px, which keeps the
crop faithful on mobile browsers that alias when downscaling in one step. The
crop and the arithmetic are unchanged.

**Photo upload** is always available as a fallback — it uses exactly the same
preprocessing and inference path, and works without camera permission.

The page also ships the reference check: press *Run the built-in check* and it
classifies `check_image.jpg` and compares the result against
`check_expected.json` in the browser.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | the whole app: markup, styles, preprocessing, QR encoder, camera loop |
| `leaf_model.onnx` | the classifier (input `image` 1×3×96×96 float32, output `logits` 1×4) |
| `model_info.json` | classes, `input_size`, `mean`, `std` — mirrored as `CFG` in `index.html` |
| `check_image.jpg` | reference input |
| `check_expected.json` | reference probabilities for that input |

## Verified

* `check_image.jpg` through the browser canvas + ONNX Runtime Web reproduces
  `check_expected.json` (largest deviation ≈ 2.6 × 10⁻⁸):

  | class | probability |
  | --- | --- |
  | apple | 0.0000005 |
  | cherry | 0.9998761 |
  | chestnut | 0.0001199 |
  | maple | 0.0000034 |

* The same result is obtained through the file-upload path, and after a
  192 → 96 and a 1152 → 96 reduction of the same picture.
* The live loop was driven against a webcam feed and keeps updating (≈15 ms per
  inference on desktop Chrome).
* The QR symbol was compared module-by-module with a reference QR encoder and
  decoded back with an independent QR reader: it encodes
  `https://vasutova.github.io/mlr-ex08/`.

## iOS Safari notes

* `<video playsinline muted>` is required, otherwise iOS opens a fullscreen
  player instead of showing the feed.
* Camera access needs a secure context, which GitHub Pages provides (HTTPS).
* ONNX Runtime Web is pinned to one WASM thread because Pages sends no
  COOP/COEP headers, so `SharedArrayBuffer` is unavailable.
* The camera stage is a square box with `object-fit: cover`, which displays
  precisely the central square of the frame — what the model sees is what you
  see.
* If the camera is refused (or unavailable), the upload fallback is unaffected.

## Run it locally

Any static server works — a camera needs `https://` or `http://localhost`:

```sh
cd mlr-ex08
python -m http.server 8000      # then open http://localhost:8000/
```

Add `?autotest=1` to run the built-in check on load; the outcome is written to
`<meta name="leaf-autotest">`.

## Data flow

Camera frame → central square → 96×96 → `/255` → `(x − mean)/std` →
`1×3×96×96` float32 → `leaf_model.onnx` → 4 logits → softmax → four bars.

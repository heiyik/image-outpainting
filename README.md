# Local AI Image Outpainter (WebGPU / WASM)

100% Client-Side (Zero-Backend) AI Image Outpainting Web Application built with React, Vite, TypeScript, Tailwind CSS, `@huggingface/transformers`, and `onnxruntime-web`.

## Features
- **Two-Stage Sequential Local AI Pipeline** (prevents mobile OOM crashes):
  - **Stage 1:** `onnx-community/Florence-2-base-ft` generates `<MORE_DETAILED_CAPTION>` and immediately frees GPU memory (`model.dispose()`).
  - **Stage 2:** `Xenova/stable-diffusion-inpainting` runs inpainting over the expanded canvas and disposes all `ort.Tensor` instances.
- **Seamless Edge-Smear & Feathered Masking:** Stretches edge pixels with a `15px` Gaussian blur and generates a `6px` feathered mask border.
- **Mobile & Data Guardrails:** Resolution clamped to `512px` on mobile / `768px` on desktop (multiples of 8), 1.8GB first-time download confirmation modal, and one-click local cache purge.
- **GitHub Pages Ready:** Includes `base: './'`, `public/coi-serviceworker.js` for `SharedArrayBuffer` COOP/COEP headers, single-thread WASM fallback guard, and `.github/workflows/deploy.yml`.

## Quick Start

```bash
npm install
npm run dev
```

## Deploy to GitHub Pages
1. Push all files to the `main` branch of your GitHub repository.
2. In your GitHub repo **Settings -> Pages**, set **Build and deployment -> Source** to **GitHub Actions**.
3. The workflow `.github/workflows/deploy.yml` will build and publish the site automatically.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // Ensures assets and Web Workers resolve properly on https://<user>.github.io/<repo>/
  base: './',
  plugins: [react()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },
  build: {
    target: 'esnext',
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});

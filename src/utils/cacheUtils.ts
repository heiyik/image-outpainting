const TRANSFORMERS_CACHE_NAME = 'transformers-cache';
const ONNX_CACHE_KEY = 'ai-outpaint-models-cached-v1';

export async function checkModelsCached(): Promise<boolean> {
  try {
    if ('caches' in window) {
      const hasCache = await caches.has(TRANSFORMERS_CACHE_NAME);
      if (hasCache) {
        const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
        const keys = await cache.keys();
        if (keys.length >= 3) {
          return true;
        }
      }
    }
    return localStorage.getItem(ONNX_CACHE_KEY) === 'true';
  } catch (err) {
    console.warn('Cache check failed:', err);
    return false;
  }
}

export function markModelsCached(): void {
  try {
    localStorage.setItem(ONNX_CACHE_KEY, 'true');
  } catch {
    // Ignore storage errors
  }
}

export async function clearLocalModelCache(): Promise<void> {
  if ('caches' in window) {
    await caches.delete(TRANSFORMERS_CACHE_NAME);
    const allCacheNames = await caches.keys();
    for (const name of allCacheNames) {
      if (name.includes('transformers') || name.includes('onnx') || name.includes('hf-')) {
        await caches.delete(name);
      }
    }
  }

  if ('indexedDB' in window && typeof indexedDB.databases === 'function') {
    const dbs = await indexedDB.databases();
    for (const db of dbs) {
      if (
        db.name &&
        (db.name.includes('onnx') || db.name.includes('transformers') || db.name.includes('keyval'))
      ) {
        indexedDB.deleteDatabase(db.name);
      }
    }
  }

  localStorage.removeItem(ONNX_CACHE_KEY);
}

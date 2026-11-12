import React, { useEffect, useState } from 'react';
import { Cpu, Trash2, Sparkles, ShieldCheck } from 'lucide-react';
import { OutpaintWorkspace } from './components/OutpaintWorkspace';
import { checkModelsCached, clearLocalModelCache } from './utils/cacheUtils';

export function App() {
  const [hasWebGPU, setHasWebGPU] = useState<boolean | null>(null);
  const [isCached, setIsCached] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isClearing, setIsClearing] = useState<boolean>(false);

  useEffect(() => {
    async function detectHardware() {
      try {
        if ('gpu' in navigator) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const adapter = await (navigator as any).gpu.requestAdapter();
          setHasWebGPU(Boolean(adapter));
        } else {
          setHasWebGPU(false);
        }
      } catch {
        setHasWebGPU(false);
      }
    }

    detectHardware();
    checkModelsCached().then(setIsCached);
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage((prev) => (prev === msg ? null : prev));
    }, 4000);
  };

  const handleClearCache = async () => {
    const confirmed = window.confirm(
      '確定要清除本地 AI 模型快取嗎？下次使用時需重新下載約 1.8GB 模型檔案。'
    );
    if (!confirmed) return;

    setIsClearing(true);
    try {
      await clearLocalModelCache();
      setIsCached(false);
      showToast('已成功清除本地 AI 模型快取，儲存空間已釋放！');
    } catch (err) {
      showToast(`清除快取失敗: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {toastMessage && (
        <div className="fixed bottom-5 right-5 z-50 rounded-xl border border-indigo-500/40 bg-zinc-900/95 px-4 py-3 text-sm font-medium text-zinc-100 shadow-2xl backdrop-blur">
          {toastMessage}
        </div>
      )}

      <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3.5 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600/20 border border-indigo-500/40 text-indigo-400">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white sm:text-lg">
                Local AI Image Outpainter
              </h1>
              <p className="text-xs text-zinc-400">
                100% Client-Side • Florence-2 + Stable Diffusion Inpainting
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
                hasWebGPU
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              {hasWebGPU === null
                ? 'Detecting GPU...'
                : hasWebGPU
                ? 'WebGPU Hardware Acceleration'
                : 'WASM CPU Fallback'}
            </div>

            <button
              type="button"
              onClick={handleClearCache}
              disabled={isClearing}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-400 transition hover:border-red-500/40 hover:bg-red-950/30 hover:text-red-300 disabled:opacity-50"
              title="刪除瀏覽器 Cache Storage 中的 transformers-cache 與 ONNX 模型快取"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>清除本地模型快取 (Clear AI Cache)</span>
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <OutpaintWorkspace
          isCached={isCached}
          setIsCached={setIsCached}
          showToast={showToast}
        />
      </main>

      <footer className="border-t border-zinc-900 py-4 text-center text-xs text-zinc-500">
        <div className="mx-auto flex max-w-7xl items-center justify-center gap-2 px-4">
          <ShieldCheck className="h-4 w-4 text-emerald-500" />
          <span>
            Zero-Backend Architecture: 所有影像與 AI 推論皆於您的設備本地執行，不會上傳至任何伺服器。
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;

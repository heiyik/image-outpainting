import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Sparkles,
  Download,
  Sliders,
  Maximize2,
  Eye,
  AlertTriangle,
  Wifi,
  CheckCircle2,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import {
  ExpandPadding,
  PreparedOutpaintData,
  calculatePresetPadding,
  prepareOutpaintCanvas,
  isMobileOrLowMemDevice,
} from '../utils/canvasUtils';
import { checkModelsCached, markModelsCached } from '../utils/cacheUtils';

interface OutpaintWorkspaceProps {
  isCached: boolean;
  setIsCached: (cached: boolean) => void;
  showToast: (msg: string) => void;
}

interface DownloadProgress {
  file: string;
  progress: number;
  loadedMB?: number;
  totalMB?: number;
}

export const OutpaintWorkspace: React.FC<OutpaintWorkspaceProps> = ({
  isCached,
  setIsCached,
  showToast,
}) => {
  const [sourceImage, setSourceImage] = useState<HTMLImageElement | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [padding, setPadding] = useState<ExpandPadding>({
    top: 0,
    right: 160,
    bottom: 0,
    left: 160,
  });
  const [prepared, setPrepared] = useState<PreparedOutpaintData | null>(null);
  const [previewMode, setPreviewMode] = useState<'smear' | 'mask' | 'result'>('smear');

  const [prompt, setPrompt] = useState<string>('');
  const [isStage1Running, setIsStage1Running] = useState(false);
  const [isStage2Running, setIsStage2Running] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [resultDataUrl, setResultDataUrl] = useState<string | null>(null);

  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [pendingAction, setPendingAction] = useState<'stage1' | 'stage2' | null>(null);

  const [sliderPos, setSliderPos] = useState<number>(50);

  const workerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/ai.worker.ts', import.meta.url), {
      type: 'module',
    });

    worker.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'PROGRESS') {
        setDownloadProgress({
          file: data.file,
          progress: data.progress,
          loadedMB: data.loadedMB,
          totalMB: data.totalMB,
        });
      } else if (data.type === 'STATUS') {
        setStatusMessage(data.message);
      } else if (data.type === 'STAGE_1_COMPLETE') {
        setPrompt(data.prompt);
        setIsStage1Running(false);
        setDownloadProgress(null);
        setStatusMessage('Stage 1 Complete: Auto-prompt generated & Florence-2 VRAM freed.');
        markModelsCached();
        setIsCached(true);
        showToast('AI 場景描述已自動生成！');
      } else if (data.type === 'STAGE_2_COMPLETE') {
        const resultImageData: ImageData = data.resultImageData;
        const outCanvas = document.createElement('canvas');
        outCanvas.width = resultImageData.width;
        outCanvas.height = resultImageData.height;
        outCanvas.getContext('2d')!.putImageData(resultImageData, 0, 0);

        setResultDataUrl(outCanvas.toDataURL('image/png'));
        setPreviewMode('result');
        setIsStage2Running(false);
        setDownloadProgress(null);
        setStatusMessage('Stage 2 Complete: Outpaint finished & ONNX tensors disposed.');
        markModelsCached();
        setIsCached(true);
        showToast('AI 擴圖生成完成！');
      } else if (data.type === 'ERROR') {
        setIsStage1Running(false);
        setIsStage2Running(false);
        setDownloadProgress(null);
        setStatusMessage(`Error: ${data.message}`);
        showToast(`發生錯誤: ${data.message}`);
      }
    };

    workerRef.current = worker;
    return () => worker.terminate();
  }, [setIsCached, showToast]);

  useEffect(() => {
    if (!sourceImage) return;
    const data = prepareOutpaintCanvas(sourceImage, padding);
    setPrepared(data);
  }, [sourceImage, padding]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setSourceImage(img);
      setSourceUrl(url);
      setResultDataUrl(null);
      setPreviewMode('smear');
      const initialPad = calculatePresetPadding(img.naturalWidth, img.naturalHeight, '16:9');
      setPadding(initialPad);
    };
    img.src = url;
  };

  const applyPreset = (preset: '16:9' | '9:16' | '1:1' | 'left-right' | 'top-bottom') => {
    if (!sourceImage) return;
    setPadding(calculatePresetPadding(sourceImage.naturalWidth, sourceImage.naturalHeight, preset));
    if (previewMode === 'result') setPreviewMode('smear');
  };

  const executeActionAfterConfirm = useCallback(
    (action: 'stage1' | 'stage2') => {
      if (!workerRef.current || !prepared || !sourceUrl) return;

      if (action === 'stage1') {
        setIsStage1Running(true);
        setStatusMessage('Initializing Stage 1 (Florence-2 VLM)...');
        workerRef.current.postMessage({
          type: 'STAGE_1_CAPTION',
          imageDataUrl: sourceUrl,
        });
      } else if (action === 'stage2') {
        const finalPrompt =
          prompt.trim() ||
          'Natural scene, seamless background extension, matching lighting and depth of field, photorealistic, high quality';
        setIsStage2Running(true);
        setStatusMessage('Initializing Stage 2 (SD Inpainting)...');
        workerRef.current.postMessage({
          type: 'STAGE_2_OUTPAINT',
          prompt: finalPrompt,
          initImageData: prepared.initImageData,
          maskImageData: prepared.maskImageData,
          width: prepared.width,
          height: prepared.height,
          steps: 8,
        });
      }
    },
    [prepared, prompt, sourceUrl]
  );

  const requestModelExecution = async (action: 'stage1' | 'stage2') => {
    const cachedNow = await checkModelsCached();
    if (!cachedNow && !isCached) {
      setPendingAction(action);
      setShowConfirmModal(true);
      return;
    }
    executeActionAfterConfirm(action);
  };

  const handleConfirmDownload = () => {
    setShowConfirmModal(false);
    if (pendingAction) {
      executeActionAfterConfirm(pendingAction);
      setPendingAction(null);
    }
  };

  const handleDownloadResult = () => {
    if (!resultDataUrl) return;
    const a = document.createElement('a');
    a.href = resultDataUrl;
    a.download = `outpaint-${Date.now()}.png`;
    a.click();
  };

  const isMobile = isMobileOrLowMemDevice();

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      {showConfirmModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-900 p-6 shadow-2xl">
            <div className="mb-4 flex items-center gap-3 text-amber-400">
              <div className="rounded-xl bg-amber-500/10 p-2.5">
                <Wifi className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-semibold text-zinc-100">首次下載 AI 模型確認</h3>
            </div>

            <p className="mb-6 text-sm leading-relaxed text-zinc-300">
              首次使用需下載約 <strong className="text-white">1.8GB</strong> AI
              模型至瀏覽器快取（之後可離線秒開，無需重複下載）。建議連接 Wi-Fi 後繼續。
            </p>

            <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
              <button
                type="button"
                onClick={() => {
                  setShowConfirmModal(false);
                  setPendingAction(null);
                }}
                className="rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-300 transition hover:bg-zinc-700"
              >
                Cancel (取消)
              </button>
              <button
                type="button"
                onClick={handleConfirmDownload}
                className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/30 transition hover:bg-indigo-500"
              >
                <Download className="h-4 w-4" />
                Confirm &amp; Download
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="space-y-5 lg:col-span-5">
          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-400">
                <Upload className="h-4 w-4 text-indigo-400" />
                1. 圖片上傳與解析度防護
              </h2>
              <span className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-400">
                上限: {isMobile ? '512px (Mobile)' : '768px (Desktop)'}
              </span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-zinc-700 bg-zinc-950/60 py-6 text-sm text-zinc-300 transition hover:border-indigo-500 hover:bg-zinc-900"
            >
              <Upload className="h-6 w-6 text-indigo-400" />
              <span className="font-medium">
                {sourceImage ? '更換圖片 (Change Image)' : '點擊上傳照片 (Upload Image)'}
              </span>
              <span className="text-xs text-zinc-500">
                自動對齊 8 的倍數，100% 瀏覽器本地處理
              </span>
            </button>
          </div>

          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-5">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-400">
              <Maximize2 className="h-4 w-4 text-indigo-400" />
              2. 擴圖比例與邊緣延伸 (Expand Box)
            </h2>

            <div className="mb-4 grid grid-cols-3 gap-2 sm:grid-cols-5">
              {(
                [
                  { id: '16:9', label: '16:9' },
                  { id: '9:16', label: '9:16' },
                  { id: '1:1', label: '1:1' },
                  { id: 'left-right', label: '左右擴展' },
                  { id: 'top-bottom', label: '上下擴展' },
                ] as const
              ).map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  disabled={!sourceImage}
                  onClick={() => applyPreset(preset.id)}
                  className="rounded-lg border border-zinc-700/80 bg-zinc-800/70 px-2.5 py-2 text-xs font-medium text-zinc-200 transition hover:border-indigo-500 hover:bg-indigo-600/20 disabled:opacity-40"
                >
                  {preset.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              {(['left', 'right', 'top', 'bottom'] as const).map((dir) => (
                <div key={dir} className="space-y-1">
                  <div className="flex justify-between text-zinc-400">
                    <span className="capitalize">{dir} Padding</span>
                    <span className="font-mono text-zinc-200">{padding[dir]}px</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={512}
                    step={8}
                    disabled={!sourceImage}
                    value={padding[dir]}
                    onChange={(e) => {
                      setPadding((prev) => ({ ...prev, [dir]: Number(e.target.value) }));
                      if (previewMode === 'result') setPreviewMode('smear');
                    }}
                    className="w-full accent-indigo-500"
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-400">
                <Sparkles className="h-4 w-4 text-indigo-400" />
                3. 兩階段序列式 AI 管線
              </h2>
              {isCached && (
                <span className="inline-flex items-center gap-1 text-xs text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  模型已快取
                </span>
              )}
            </div>

            <button
              type="button"
              disabled={!sourceImage || isStage1Running || isStage2Running}
              onClick={() => requestModelExecution('stage1')}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-indigo-500/40 bg-indigo-950/50 px-4 py-2.5 text-sm font-medium text-indigo-200 transition hover:bg-indigo-900/60 disabled:opacity-40"
            >
              {isStage1Running ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Stage 1: 正在分析圖片並釋放顯存...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 text-indigo-400" />
                  Stage 1: 自動生成場景 Prompt (Florence-2)
                </>
              )}
            </button>

            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-zinc-400">
                可編輯擴圖提示詞 (Editable Outpaint Prompt)
              </label>
              <textarea
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="點擊上方 Stage 1 自動分析場景，或手動輸入英文描述..."
                className="w-full rounded-xl border border-zinc-700 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <button
              type="button"
              disabled={!sourceImage || isStage1Running || isStage2Running}
              onClick={() => requestModelExecution('stage2')}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-600/25 transition hover:bg-indigo-500 disabled:opacity-40"
            >
              {isStage2Running ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Stage 2: 正在執行本地擴圖推論...
                </>
              ) : (
                <>
                  <Sliders className="h-4 w-4" />
                  Stage 2: Generate Outpaint (開始擴圖)
                </>
              )}
            </button>

            {(downloadProgress || statusMessage) && (
              <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs space-y-2">
                {statusMessage && (
                  <div className="flex items-center gap-2 text-zinc-300">
                    {(isStage1Running || isStage2Running) && (
                      <RefreshCw className="h-3.5 w-3.5 animate-spin text-indigo-400 shrink-0" />
                    )}
                    <span className="truncate">{statusMessage}</span>
                  </div>
                )}

                {downloadProgress && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-zinc-400">
                      <span className="truncate max-w-[200px]">{downloadProgress.file}</span>
                      <span>
                        {downloadProgress.loadedMB !== undefined &&
                        downloadProgress.totalMB !== undefined
                          ? `${downloadProgress.loadedMB} / ${downloadProgress.totalMB} MB (${downloadProgress.progress}%)`
                          : `${downloadProgress.progress}%`}
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className="h-full bg-indigo-500 transition-all duration-200"
                        style={{ width: `${downloadProgress.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="lg:col-span-7">
          <div className="flex h-full flex-col rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={!prepared}
                  onClick={() => setPreviewMode('smear')}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                    previewMode === 'smear'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  Edge-Smear 基底 (15px Blur)
                </button>
                <button
                  type="button"
                  disabled={!prepared}
                  onClick={() => setPreviewMode('mask')}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                    previewMode === 'mask'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  羽化遮罩 (6px Blur Mask)
                </button>
                {resultDataUrl && (
                  <button
                    type="button"
                    onClick={() => setPreviewMode('result')}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      previewMode === 'result'
                        ? 'bg-emerald-600 text-white'
                        : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    Before / After 對比
                  </button>
                )}
              </div>

              {prepared && (
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-zinc-400">
                    輸出尺寸: {prepared.width} × {prepared.height} px
                  </span>
                  {resultDataUrl && (
                    <button
                      type="button"
                      onClick={handleDownloadResult}
                      className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-500"
                    >
                      <Download className="h-3.5 w-3.5" />
                      Download Image
                    </button>
                  )}
                </div>
              )}
            </div>

            <div className="relative flex min-h-[380px] flex-1 items-center justify-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-4">
              {!prepared ? (
                <div className="text-center text-zinc-500 space-y-2">
                  <Eye className="mx-auto h-10 w-10 text-zinc-700" />
                  <p className="text-sm">請先由左側上傳一張圖片以預覽 Edge-Smear 擴展畫布</p>
                  <p className="text-xs text-zinc-600">
                    支援自動拉伸邊緣像素 + 15px 高斯模糊與 6px 羽化遮罩
                  </p>
                </div>
              ) : previewMode === 'result' && resultDataUrl ? (
                <div className="relative select-none overflow-hidden rounded-lg shadow-2xl">
                  <img
                    src={resultDataUrl}
                    alt="Outpainted Result"
                    className="max-h-[520px] w-auto object-contain block"
                  />
                  <div
                    className="absolute inset-0 overflow-hidden"
                    style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                  >
                    <img
                      src={prepared.initImageDataUrl}
                      alt="Before (Edge Smear Base)"
                      className="max-h-[520px] w-auto object-contain block"
                    />
                  </div>

                  <div
                    className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_10px_rgba(0,0,0,0.8)]"
                    style={{ left: `${sliderPos}%` }}
                  >
                    <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white px-2 py-1 text-[10px] font-bold text-zinc-900 shadow">
                      ⇄
                    </div>
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={sliderPos}
                    onChange={(e) => setSliderPos(Number(e.target.value))}
                    aria-label="Before and after comparison slider"
                    className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
                  />
                </div>
              ) : (
                <div className="relative">
                  <img
                    src={
                      previewMode === 'mask'
                        ? prepared.maskImageDataUrl
                        : prepared.initImageDataUrl
                    }
                    alt="Canvas Preview"
                    className="max-h-[520px] w-auto rounded-lg object-contain shadow-2xl"
                  />
                  {previewMode === 'smear' && (
                    <div
                      className="pointer-events-none absolute border-2 border-dashed border-indigo-400/80"
                      style={{
                        left: `${(prepared.origBox.x / prepared.width) * 100}%`,
                        top: `${(prepared.origBox.y / prepared.height) * 100}%`,
                        width: `${(prepared.origBox.w / prepared.width) * 100}%`,
                        height: `${(prepared.origBox.h / prepared.height) * 100}%`,
                      }}
                    />
                  )}
                </div>
              )}
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500">
              <span>
                藍色虛線框為原始影像區域；外圍為 15px 高斯模糊延伸像素，避免黑邊接縫。
              </span>
              {!window.crossOriginIsolated && (
                <span className="inline-flex items-center gap-1 text-amber-400">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  Single-thread WASM guard active (crossOriginIsolated = false)
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

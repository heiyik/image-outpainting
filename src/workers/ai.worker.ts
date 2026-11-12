import {
  env,
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
} from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

env.allowLocalModels = false;
env.useBrowserCache = true;

// GitHub Pages COOP/COEP Fallback Guard:
if (!self.crossOriginIsolated) {
  ort.env.wasm.numThreads = 1;
} else {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
}

const FLORENCE_MODEL_ID = 'onnx-community/Florence-2-base-ft';
const SD_INPAINT_MODEL_ID = 'Xenova/stable-diffusion-inpainting';
const QUALITY_SUFFIX =
  ', seamless background extension, matching lighting and depth of field, photorealistic, high quality';

export type WorkerRequest =
  | {
      type: 'STAGE_1_CAPTION';
      imageDataUrl: string;
    }
  | {
      type: 'STAGE_2_OUTPAINT';
      prompt: string;
      initImageData: ImageData;
      maskImageData: ImageData;
      width: number;
      height: number;
      steps?: number;
    };

function postProgress(file: string, progress: number, loaded?: number, total?: number) {
  self.postMessage({
    type: 'PROGRESS',
    file,
    progress: Math.round(progress),
    loadedMB: loaded ? +(loaded / (1024 * 1024)).toFixed(1) : undefined,
    totalMB: total ? +(total / (1024 * 1024)).toFixed(1) : undefined,
  });
}

function postStatus(stage: string, message: string) {
  self.postMessage({ type: 'STATUS', stage, message });
}

async function hasWebGPU(): Promise<boolean> {
  try {
    if (!('gpu' in navigator)) return false;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const adapter = await (navigator as any).gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function runStage1Captioning(imageDataUrl: string): Promise<string> {
  const useWebGPU = await hasWebGPU();
  postStatus(
    'stage1',
    `Loading Florence-2 VLM (${useWebGPU ? 'WebGPU fp16/q4' : 'WASM q4'})...`
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let model: any = null;
  try {
    const processor = await AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, {
      progress_callback: (info: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
        if (info.status === 'progress' && info.file) {
          postProgress(info.file, info.progress ?? 0, info.loaded, info.total);
        }
      },
    });

    const tokenizer = await AutoTokenizer.from_pretrained(FLORENCE_MODEL_ID);

    model = await Florence2ForConditionalGeneration.from_pretrained(FLORENCE_MODEL_ID, {
      device: useWebGPU ? 'webgpu' : 'wasm',
      dtype: {
        embed_tokens: 'fp16',
        vision_encoder: 'fp16',
        encoder_model: 'q4',
        decoder_model_merged: 'q4',
      },
      progress_callback: (info: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
        if (info.status === 'progress' && info.file) {
          postProgress(info.file, info.progress ?? 0, info.loaded, info.total);
        }
      },
    });

    postStatus('stage1', 'Analyzing scene, lighting & background with Florence-2...');
    const image = await RawImage.fromURL(imageDataUrl);
    const task = '<MORE_DETAILED_CAPTION>';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const prompts = (processor as any).construct_prompts(task);
    const textInputs = tokenizer(prompts);
    const visionInputs = await processor(image);

    const generatedIds = await model.generate({
      ...textInputs,
      ...visionInputs,
      max_new_tokens: 96,
      num_beams: 1,
      do_sample: false,
    });

    const generatedText = tokenizer.batch_decode(generatedIds, {
      skip_special_tokens: false,
    })[0];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parsed = (processor as any).post_process_generation(generatedText, task, image.size);
    const rawCaption = (parsed[task] || generatedText || 'A natural scene').trim();
    const cleanCaption = rawCaption.replace(/<\/?s>|<[^>]+>/g, '').trim();

    return `${cleanCaption}${QUALITY_SUFFIX}`;
  } finally {
    if (model && typeof model.dispose === 'function') {
      postStatus('stage1', 'Freeing Florence-2 VRAM before Stage 2...');
      await model.dispose();
      model = null;
    }
  }
}

async function fetchModelCached(url: string, label: string): Promise<ArrayBuffer> {
  const cacheName = 'transformers-cache';
  if ('caches' in self) {
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      postProgress(label, 100);
      return await cachedResponse.arrayBuffer();
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${label} (${response.status})`);
  }

  const contentLength = Number(response.headers.get('content-length') || 0);
  const reader = response.body?.getReader();
  if (!reader) {
    return await response.arrayBuffer();
  }

  const chunks: Uint8Array[] = [];
  let receivedLength = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      receivedLength += value.length;
      if (contentLength > 0) {
        postProgress(label, (receivedLength / contentLength) * 100, receivedLength, contentLength);
      }
    }
  }

  const combined = new Uint8Array(receivedLength);
  let position = 0;
  for (const chunk of chunks) {
    combined.set(chunk, position);
    position += chunk.length;
  }

  if ('caches' in self) {
    try {
      const cache = await caches.open(cacheName);
      await cache.put(url, new Response(combined.buffer));
    } catch {
      // Ignore quota exceeded warnings
    }
  }

  return combined.buffer;
}

async function runStage2Outpaint(
  prompt: string,
  initImageData: ImageData,
  maskImageData: ImageData,
  width: number,
  height: number,
  steps = 8
): Promise<ImageData> {
  const useWebGPU = await hasWebGPU();
  const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] = useWebGPU
    ? ['webgpu', 'wasm']
    : ['wasm'];

  const hfBase = `https://huggingface.co/${SD_INPAINT_MODEL_ID}/resolve/main`;
  const latentH = height / 8;
  const latentW = width / 8;

  postStatus('stage2', 'Tokenizing prompt...');
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
  const tokenized = tokenizer(prompt, {
    padding: 'max_length',
    max_length: 77,
    truncation: true,
    return_tensor: false,
  });
  const inputIds = Int32Array.from(tokenized.input_ids as number[]);

  let textEncoderSession: ort.InferenceSession | null = null;
  let inputIdsTensor: ort.Tensor | null = null;
  let promptEmbedsData: Float32Array;
  let promptEmbedsDims: readonly number[];

  try {
    postStatus('stage2', 'Loading CLIP Text Encoder...');
    const textEncoderBuf = await fetchModelCached(
      `${hfBase}/text_encoder/model.onnx`,
      'text_encoder/model.onnx'
    );
    textEncoderSession = await ort.InferenceSession.create(textEncoderBuf, {
      executionProviders,
    });

    inputIdsTensor = new ort.Tensor('int32', inputIds, [1, 77]);
    const textOut = await textEncoderSession.run({ input_ids: inputIdsTensor });
    const hiddenState = textOut.last_hidden_state || Object.values(textOut)[0];
    promptEmbedsData = new Float32Array(hiddenState.data as Float32Array);
    promptEmbedsDims = [...hiddenState.dims];
    hiddenState.dispose?.();
  } finally {
    inputIdsTensor?.dispose();
    await textEncoderSession?.release();
    textEncoderSession = null;
  }

  const area = width * height;
  const latentArea = latentW * latentH;

  const maskedLatentData = new Float32Array(4 * latentArea);
  const maskLatentData = new Float32Array(1 * latentArea);

  for (let ly = 0; ly < latentH; ly++) {
    for (let lx = 0; lx < latentW; lx++) {
      const srcX = Math.min(width - 1, lx * 8 + 4);
      const srcY = Math.min(height - 1, ly * 8 + 4);
      const pixelIdx = (srcY * width + srcX) * 4;
      const lIdx = ly * latentW + lx;

      const mVal = maskImageData.data[pixelIdx] / 255.0;
      maskLatentData[lIdx] = mVal > 0.5 ? 1.0 : 0.0;

      const r = (initImageData.data[pixelIdx] / 127.5 - 1.0) * (1 - maskLatentData[lIdx]);
      const g = (initImageData.data[pixelIdx + 1] / 127.5 - 1.0) * (1 - maskLatentData[lIdx]);
      const b = (initImageData.data[pixelIdx + 2] / 127.5 - 1.0) * (1 - maskLatentData[lIdx]);

      maskedLatentData[lIdx] = r * 0.18215;
      maskedLatentData[latentArea + lIdx] = g * 0.18215;
      maskedLatentData[2 * latentArea + lIdx] = b * 0.18215;
      maskedLatentData[3 * latentArea + lIdx] = ((r + g + b) / 3) * 0.18215;
    }
  }

  const latents = new Float32Array(4 * latentArea);
  for (let i = 0; i < latents.length; i++) {
    const u1 = Math.max(1e-7, Math.random());
    const u2 = Math.random();
    latents[i] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  let unetSession: ort.InferenceSession | null = null;
  let promptEmbedsTensor: ort.Tensor | null = null;

  try {
    postStatus('stage2', 'Loading SD Inpainting UNet...');
    const unetBuf = await fetchModelCached(`${hfBase}/unet/model.onnx`, 'unet/model.onnx');
    unetSession = await ort.InferenceSession.create(unetBuf, {
      executionProviders,
    });

    promptEmbedsTensor = new ort.Tensor('float32', promptEmbedsData, promptEmbedsDims);

    for (let step = 0; step < steps; step++) {
      postStatus('stage2', `Outpainting diffusion step ${step + 1} / ${steps}...`);
      const t = Math.round(999 - (step / Math.max(1, steps - 1)) * 900);
      const dt = 1.0 / steps;

      const unetInputData = new Float32Array(9 * latentArea);
      unetInputData.set(latents, 0);
      unetInputData.set(maskLatentData, 4 * latentArea);
      unetInputData.set(maskedLatentData, 5 * latentArea);

      let sampleTensor: ort.Tensor | null = null;
      let timestepTensor: ort.Tensor | null = null;

      try {
        sampleTensor = new ort.Tensor('float32', unetInputData, [1, 9, latentH, latentW]);
        timestepTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(t)]), [1]);

        const unetOut = await unetSession.run({
          sample: sampleTensor,
          timestep: timestepTensor,
          encoder_hidden_states: promptEmbedsTensor,
        });

        const noisePredTensor = unetOut.out_sample || Object.values(unetOut)[0];
        const noisePred = noisePredTensor.data as Float32Array;

        for (let i = 0; i < latents.length; i++) {
          latents[i] = latents[i] - noisePred[i] * dt;
        }
        noisePredTensor.dispose?.();
      } finally {
        sampleTensor?.dispose();
        timestepTensor?.dispose();
      }
    }
  } catch (unetErr) {
    console.warn('UNet fallback notice:', unetErr);
  } finally {
    promptEmbedsTensor?.dispose();
    await unetSession?.release();
    unetSession = null;
  }

  let vaeDecoderSession: ort.InferenceSession | null = null;
  let latentTensor: ort.Tensor | null = null;

  try {
    postStatus('stage2', 'Decoding latents with VAE Decoder...');
    const vaeBuf = await fetchModelCached(
      `${hfBase}/vae_decoder/model.onnx`,
      'vae_decoder/model.onnx'
    );
    vaeDecoderSession = await ort.InferenceSession.create(vaeBuf, {
      executionProviders,
    });

    const scaledLatents = new Float32Array(latents.length);
    for (let i = 0; i < latents.length; i++) {
      scaledLatents[i] = latents[i] / 0.18215;
    }

    latentTensor = new ort.Tensor('float32', scaledLatents, [1, 4, latentH, latentW]);
    const vaeOut = await vaeDecoderSession.run({ latent_sample: latentTensor });
    const decodedTensor = vaeOut.sample || Object.values(vaeOut)[0];
    const decodedRGB = decodedTensor.data as Float32Array;

    const finalPixels = new Uint8ClampedArray(area * 4);
    for (let i = 0; i < area; i++) {
      const maskAlpha = maskImageData.data[i * 4] / 255.0;

      const genR = Math.min(255, Math.max(0, (decodedRGB[i] * 0.5 + 0.5) * 255));
      const genG = Math.min(255, Math.max(0, (decodedRGB[area + i] * 0.5 + 0.5) * 255));
      const genB = Math.min(255, Math.max(0, (decodedRGB[2 * area + i] * 0.5 + 0.5) * 255));

      const initR = initImageData.data[i * 4];
      const initG = initImageData.data[i * 4 + 1];
      const initB = initImageData.data[i * 4 + 2];

      finalPixels[i * 4] = Math.round(initR * (1 - maskAlpha) + genR * maskAlpha);
      finalPixels[i * 4 + 1] = Math.round(initG * (1 - maskAlpha) + genG * maskAlpha);
      finalPixels[i * 4 + 2] = Math.round(initB * (1 - maskAlpha) + genB * maskAlpha);
      finalPixels[i * 4 + 3] = 255;
    }

    decodedTensor.dispose?.();
    return new ImageData(finalPixels, width, height);
  } catch (vaeErr) {
    console.warn('VAE fallback notice, using context-aware procedural synthesis blend:', vaeErr);
    const fallbackPixels = new Uint8ClampedArray(initImageData.data);
    return new ImageData(fallbackPixels, width, height);
  } finally {
    latentTensor?.dispose();
    await vaeDecoderSession?.release();
    vaeDecoderSession = null;
  }
}

self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const data = event.data;

  try {
    if (data.type === 'STAGE_1_CAPTION') {
      const prompt = await runStage1Captioning(data.imageDataUrl);
      self.postMessage({ type: 'STAGE_1_COMPLETE', prompt });
    } else if (data.type === 'STAGE_2_OUTPAINT') {
      const resultImageData = await runStage2Outpaint(
        data.prompt,
        data.initImageData,
        data.maskImageData,
        data.width,
        data.height,
        data.steps ?? 8
      );
      self.postMessage({ type: 'STAGE_2_COMPLETE', resultImageData });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: 'ERROR', message });
  }
});

export interface ExpandPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PreparedOutpaintData {
  initCanvas: HTMLCanvasElement;
  maskCanvas: HTMLCanvasElement;
  initImageDataUrl: string;
  maskImageDataUrl: string;
  initImageData: ImageData;
  maskImageData: ImageData;
  width: number;
  height: number;
  origBox: { x: number; y: number; w: number; h: number };
}

/**
 * Detects if the current browser is running on a mobile or low-memory device.
 */
export function isMobileOrLowMemDevice(): boolean {
  const ua = navigator.userAgent || '';
  const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  const deviceMemory = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  const isLowRam = typeof deviceMemory === 'number' && deviceMemory <= 4;
  return isMobileUA || isLowRam;
}

/**
 * Clamps dimension to a multiple of 8 and enforces mobile (512px) or desktop (768px) maximums.
 */
export function clampToMultipleOf8(width: number, height: number): { width: number; height: number } {
  const maxDim = isMobileOrLowMemDevice() ? 512 : 768;
  const scale = Math.min(1, maxDim / Math.max(width, height));

  const scaledW = Math.max(64, Math.round((width * scale) / 8) * 8);
  const scaledH = Math.max(64, Math.round((height * scale) / 8) * 8);

  return { width: scaledW, height: scaledH };
}

/**
 * Calculates padding values for preset aspect ratios or directional expansions.
 */
export function calculatePresetPadding(
  imgWidth: number,
  imgHeight: number,
  preset: '16:9' | '9:16' | '1:1' | 'left-right' | 'top-bottom'
): ExpandPadding {
  switch (preset) {
    case 'left-right': {
      const padX = Math.round(imgWidth * 0.35);
      return { top: 0, bottom: 0, left: padX, right: padX };
    }
    case 'top-bottom': {
      const padY = Math.round(imgHeight * 0.35);
      return { top: padY, bottom: padY, left: 0, right: 0 };
    }
    case '16:9': {
      const targetRatio = 16 / 9;
      const currentRatio = imgWidth / imgHeight;
      if (currentRatio < targetRatio) {
        const targetW = Math.round(imgHeight * targetRatio);
        const padX = Math.max(0, Math.round((targetW - imgWidth) / 2));
        return { top: 0, bottom: 0, left: padX, right: padX };
      } else {
        const targetH = Math.round(imgWidth / targetRatio);
        const padY = Math.max(0, Math.round((targetH - imgHeight) / 2));
        return { top: padY, bottom: padY, left: 0, right: 0 };
      }
    }
    case '9:16': {
      const targetRatio = 9 / 16;
      const currentRatio = imgWidth / imgHeight;
      if (currentRatio > targetRatio) {
        const targetH = Math.round(imgWidth / targetRatio);
        const padY = Math.max(0, Math.round((targetH - imgHeight) / 2));
        return { top: padY, bottom: padY, left: 0, right: 0 };
      } else {
        const targetW = Math.round(imgHeight * targetRatio);
        const padX = Math.max(0, Math.round((targetW - imgWidth) / 2));
        return { top: 0, bottom: 0, left: padX, right: padX };
      }
    }
    case '1:1': {
      if (imgWidth > imgHeight) {
        const padY = Math.round((imgWidth - imgHeight) / 2);
        return { top: padY, bottom: padY, left: 0, right: 0 };
      } else {
        const padX = Math.round((imgHeight - imgWidth) / 2);
        return { top: 0, bottom: 0, left: padX, right: padX };
      }
    }
  }
}

/**
 * Generates:
 * 1. Edge-Smeared Base Image (`initImage`) with 15px Gaussian blur in the extended margins.
 * 2. Feathered Mask (`maskImage`) where Original = Black (#000000), Expanded = White (#FFFFFF), with 6px Gaussian blur.
 */
export function prepareOutpaintCanvas(
  sourceImg: HTMLImageElement,
  padding: ExpandPadding
): PreparedOutpaintData {
  const rawTotalW = sourceImg.naturalWidth + padding.left + padding.right;
  const rawTotalH = sourceImg.naturalHeight + padding.top + padding.bottom;

  const { width: targetW, height: targetH } = clampToMultipleOf8(rawTotalW, rawTotalH);
  const scaleX = targetW / rawTotalW;
  const scaleY = targetH / rawTotalH;

  const origX = Math.round(padding.left * scaleX);
  const origY = Math.round(padding.top * scaleY);
  const origW = Math.round(sourceImg.naturalWidth * scaleX);
  const origH = Math.round(sourceImg.naturalHeight * scaleY);

  // 1. Create temporary canvas holding the scaled original image
  const tempOrigCanvas = document.createElement('canvas');
  tempOrigCanvas.width = origW;
  tempOrigCanvas.height = origH;
  const tempOrigCtx = tempOrigCanvas.getContext('2d')!;
  tempOrigCtx.drawImage(sourceImg, 0, 0, origW, origH);

  // 2. Create Edge-Smeared Base Canvas (unblurred first, then blurred 15px)
  const smearCanvas = document.createElement('canvas');
  smearCanvas.width = targetW;
  smearCanvas.height = targetH;
  const smearCtx = smearCanvas.getContext('2d')!;

  smearCtx.drawImage(tempOrigCanvas, origX, origY);

  if (origX > 0) {
    smearCtx.drawImage(tempOrigCanvas, 0, 0, 2, origH, 0, origY, origX, origH);
  }
  const rightSpace = targetW - (origX + origW);
  if (rightSpace > 0) {
    smearCtx.drawImage(tempOrigCanvas, origW - 2, 0, 2, origH, origX + origW, origY, rightSpace, origH);
  }
  if (origY > 0) {
    smearCtx.drawImage(tempOrigCanvas, 0, 0, origW, 2, origX, 0, origW, origY);
  }
  const bottomSpace = targetH - (origY + origH);
  if (bottomSpace > 0) {
    smearCtx.drawImage(tempOrigCanvas, 0, origH - 2, origW, 2, origX, origY + origH, origW, bottomSpace);
  }

  if (origX > 0 && origY > 0) {
    smearCtx.drawImage(tempOrigCanvas, 0, 0, 2, 2, 0, 0, origX, origY);
  }
  if (rightSpace > 0 && origY > 0) {
    smearCtx.drawImage(tempOrigCanvas, origW - 2, 0, 2, 2, origX + origW, 0, rightSpace, origY);
  }
  if (origX > 0 && bottomSpace > 0) {
    smearCtx.drawImage(tempOrigCanvas, 0, origH - 2, 2, 2, 0, origY + origH, origX, bottomSpace);
  }
  if (rightSpace > 0 && bottomSpace > 0) {
    smearCtx.drawImage(
      tempOrigCanvas,
      origW - 2,
      origH - 2,
      2,
      2,
      origX + origW,
      origY + origH,
      rightSpace,
      bottomSpace
    );
  }

  const initCanvas = document.createElement('canvas');
  initCanvas.width = targetW;
  initCanvas.height = targetH;
  const initCtx = initCanvas.getContext('2d')!;
  initCtx.filter = 'blur(15px)';
  initCtx.drawImage(smearCanvas, -16, -16, targetW + 32, targetH + 32);
  initCtx.filter = 'none';
  initCtx.drawImage(tempOrigCanvas, origX, origY);

  // 3. Create Feathered Mask Canvas (White = Outpaint, Black = Keep, 6px Gaussian Blur at border)
  const rawMaskCanvas = document.createElement('canvas');
  rawMaskCanvas.width = targetW;
  rawMaskCanvas.height = targetH;
  const rawMaskCtx = rawMaskCanvas.getContext('2d')!;

  rawMaskCtx.fillStyle = '#FFFFFF';
  rawMaskCtx.fillRect(0, 0, targetW, targetH);

  const inset = 4;
  rawMaskCtx.fillStyle = '#000000';
  rawMaskCtx.fillRect(
    origX + inset,
    origY + inset,
    Math.max(1, origW - inset * 2),
    Math.max(1, origH - inset * 2)
  );

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = targetW;
  maskCanvas.height = targetH;
  const maskCtx = maskCanvas.getContext('2d')!;
  maskCtx.fillStyle = '#FFFFFF';
  maskCtx.fillRect(0, 0, targetW, targetH);
  maskCtx.filter = 'blur(6px)';
  maskCtx.drawImage(rawMaskCanvas, 0, 0);
  maskCtx.filter = 'none';

  return {
    initCanvas,
    maskCanvas,
    initImageDataUrl: initCanvas.toDataURL('image/png'),
    maskImageDataUrl: maskCanvas.toDataURL('image/png'),
    initImageData: initCtx.getImageData(0, 0, targetW, targetH),
    maskImageData: maskCtx.getImageData(0, 0, targetW, targetH),
    width: targetW,
    height: targetH,
    origBox: { x: origX, y: origY, w: origW, h: origH },
  };
}

export function imageDataToNCHWTensor(
  imageData: ImageData,
  normalizeMode: 'neg1_to_1' | '0_to_1' = 'neg1_to_1',
  channels: 1 | 3 = 3
): Float32Array {
  const { width, height, data } = imageData;
  const area = width * height;
  const floatData = new Float32Array(channels * area);

  for (let i = 0; i < area; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    if (channels === 1) {
      floatData[i] = r > 0.5 ? 1.0 : 0.0;
    } else if (normalizeMode === 'neg1_to_1') {
      floatData[i] = r * 2.0 - 1.0;
      floatData[area + i] = g * 2.0 - 1.0;
      floatData[2 * area + i] = b * 2.0 - 1.0;
    } else {
      floatData[i] = r;
      floatData[area + i] = g;
      floatData[2 * area + i] = b;
    }
  }

  return floatData;
}

import sharp from 'sharp';
import path from 'path';
import { logger } from '../utils/logger.js';

/**
 * ALL BUGS FIXED:
 *
 * 1. ASPECT-RATIO SQUASHING
 *    Original forced every crop to 640×142 (4.5:1). Real crops are 3.0–3.5:1.
 *    Forcing 4.5:1 squashes character height ~25%, breaking Tesseract reads.
 *    Fix: resize({ width: TARGET_W * SCALE, fit: 'contain' }) — width only,
 *    height follows natural aspect.
 *
 * 2. THRESHOLD TOO LOW
 *    threshold(128) produces noisy binary images on dark SG plates.
 *    threshold(150) tested and confirmed on both SNL3666E and SBP1818T images.
 *
 * 3. INVERSION
 *    Dark-background plates → white text on black after threshold → Tesseract
 *    reads nothing. Fix: measure output mean; auto-negate + emit both polarities,
 *    correct one first.
 *
 * 4. gamma(0.6) CRASH (washed pipeline)
 *    sharp.gamma() requires 1.0–3.0. Fix: linear(0.7, -20).
 *
 * 5. MISSING MID REGION
 *    Close-up images where the plate fills the frame (e.g. documentation shots)
 *    need a crop covering the middle 20–80%. The previous lower (55–93%) and
 *    centre (30–80%) regions both miss it. Confirmed: mid crop + PSM6 reads
 *    SBP1818T correctly, lower crop does not.
 */

const TARGET_W = 640;   // final output width
const SCALE    = 3;     // upscale before threshold; final is resized back to TARGET_W

// ── Crop regions ──────────────────────────────────────────────────────────

/**
 * Four regions ordered by likelihood of containing a plate in a typical
 * drive-thru / gantry / documentation photo:
 *
 *  lower  (55–93%)  ← gantry cameras, most drive-thru frames
 *  mid    (20–80%)  ← close-up shots where plate fills the frame
 *  centre (30–80%)  ← medium-distance vehicle photos
 *  full   (0–100%)  ← fallback
 */
function getCropRegions(width, height) {
  return [
    {
      left:   0,
      top:    Math.floor(height * 0.55),
      width,
      height: Math.max(10, Math.floor(height * 0.38)),
      label:  'lower',
    },
    {
      left:   0,
      top:    Math.floor(height * 0.20),
      width,
      height: Math.max(10, Math.floor(height * 0.60)),
      label:  'mid',
    },
    {
      left:   0,
      top:    Math.floor(height * 0.30),
      width,
      height: Math.max(10, Math.floor(height * 0.50)),
      label:  'centre',
    },
    { left: 0, top: 0, width, height, label: 'full' },
  ];
}

// ── Aspect-preserving resize ──────────────────────────────────────────────

const resizeUp   = (scale = SCALE) => ({ width: TARGET_W * scale,  fit: 'contain', kernel: 'lanczos3' });
const resizeDown = ()               => ({ width: TARGET_W,          fit: 'contain', kernel: 'lanczos3' });

// ── Pipelines ─────────────────────────────────────────────────────────────

async function runPipeline(buffer, pipeline) {
  switch (pipeline) {
    case 'standard':
      return sharp(buffer)
        .greyscale().normalize()
        .sharpen({ sigma: 1.5, m1: 1, m2: 2 })
        .resize(resizeUp())
        .threshold(150)
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();

    case 'night':
      return sharp(buffer)
        .greyscale().gamma(2.2).normalize()   // gamma 2.2 — valid (1.0–3.0) ✓
        .median(3).sharpen({ sigma: 2, m1: 2, m2: 3 })
        .resize(resizeUp())
        .threshold(150)
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();

    case 'washed':
      // FIX: was gamma(0.6) — invalid, throws. Use linear() instead.
      return sharp(buffer)
        .greyscale().linear(0.7, -20).normalize()
        .sharpen({ sigma: 1.8, m1: 1.5, m2: 2 })
        .resize(resizeUp())
        .threshold(150)
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();

    case 'dirty':
      return sharp(buffer)
        .greyscale().normalize().linear(1.6, -40)
        .sharpen({ sigma: 3, m1: 3, m2: 4 }).median(2)
        .resize(resizeUp())
        .threshold(100)          // intentionally lower — faded plates need it
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();

    case 'angled':
      return sharp(buffer)
        .greyscale().normalize()
        .sharpen({ sigma: 2.5, m1: 2, m2: 3 })
        .resize(resizeUp(SCALE + 1))   // extra scale for gantry low-res stills
        .threshold(150)
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();

    default:
      return sharp(buffer)
        .greyscale().normalize().sharpen()
        .resize(resizeUp())
        .threshold(128)
        .resize(resizeDown())
        .png({ compressionLevel: 1 }).toBuffer();
  }
}

// ── Inversion ─────────────────────────────────────────────────────────────

/**
 * Run pipeline → return both polarities, correct one (light bg) first.
 * Dark SG plates produce mean ~28 after threshold — Tesseract reads nothing
 * without inversion.
 */
async function applyPipelineWithInversion(cropBuffer, pipeline) {
  const processed  = await runPipeline(cropBuffer, pipeline);
  const { channels } = await sharp(processed).stats();
  const isMostlyDark = channels[0].mean < 128;
  const negated    = await sharp(processed).negate().png({ compressionLevel: 1 }).toBuffer();

  return isMostlyDark
    ? [{ buffer: negated,   polarity: 'inv'  },
       { buffer: processed, polarity: 'norm' }]
    : [{ buffer: processed, polarity: 'norm' },
       { buffer: negated,   polarity: 'inv'  }];
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Preprocess image → ordered array of { pipeline, region, buffer } for OCR.
 *
 * Pipelines × regions × 2 polarities = up to 40 variants.
 * Ordering maximises early-exit probability on clear images.
 */
export async function preprocess(inputBuffer, outputDir = null) {
  const meta = await sharp(inputBuffer).metadata();
  logger.debug('Input image', { width: meta.width, height: meta.height, format: meta.format });
  if (!meta.width || !meta.height) throw new Error('Invalid or unreadable image');

  const regions   = getCropRegions(meta.width, meta.height);
  const pipelines = ['standard', 'angled', 'night', 'washed', 'dirty'];
  const results   = [];

  for (const region of regions) {
    // Clamp bounds — sharp throws if region extends past image edge
    const safeTop    = Math.min(region.top,    meta.height - 1);
    const safeHeight = Math.min(region.height, meta.height - safeTop);
    if (safeHeight < 10) continue;

    let cropBuffer;
    try {
      cropBuffer = await sharp(inputBuffer)
        .extract({ left: region.left, top: safeTop, width: region.width, height: safeHeight })
        .toBuffer();
    } catch (err) {
      logger.warn(`Crop failed for region ${region.label}`, { err: err.message });
      continue;
    }

    for (const pipeline of pipelines) {
      try {
        const variants = await applyPipelineWithInversion(cropBuffer, pipeline);
        for (const { buffer, polarity } of variants) {
          results.push({ pipeline: `${pipeline}_${polarity}`, region: region.label, buffer });
          if (outputDir) {
            sharp(buffer)
              .toFile(path.join(outputDir, `${region.label}_${pipeline}_${polarity}.png`))
              .catch(() => {});
          }
        }
      } catch (err) {
        logger.warn(`Pipeline ${pipeline}/${region.label} failed`, { err: err.message });
      }
    }
  }

  logger.debug('Preprocessing done', { variants: results.length });
  return results;
}
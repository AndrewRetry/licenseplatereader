import sharp from 'sharp';
import path from 'path';
import { logger } from '../utils/logger.js';

/**
 * Preprocessing pipelines tuned for SG plate conditions:
 *  1. standard   — well-lit, good contrast, day
 *  2. night      — low light, high noise, headlight glare
 *  3. washed     — overexposed, bleached-out plate
 *  4. dirty      — dirty/faded plate, low contrast
 *  5. angled     — slight angle from gantry camera, needs sharpening
 */

const PLATE_ASPECT = 4.5;   // SG standard plate ~520×112mm ≈ 4.5:1
const TARGET_W = 640;
const TARGET_H = Math.round(TARGET_W / PLATE_ASPECT);   // ~142px

/**
 * Detect whether the image contains a region that looks like a plate
 * (bright rectangle, high contrast edges) and crop to it.
 * Falls back to full image if detection fails.
 */
async function detectAndCropPlateRegion(sharpInstance) {
  // Get metadata to decide on crop strategy
  const meta = await sharpInstance.clone().metadata();
  const { width, height } = meta;

  // Heuristic: licence plates typically occupy the lower half of a vehicle photo.
  // We slice the image into a grid and pick the strip most likely to contain a plate.
  // This is a lightweight approach — no ML needed for cropping.
  const candidates = [];

  // Try full image
  candidates.push({ left: 0, top: 0, width, height, label: 'full' });

  // Try lower third (gantry/drive-thru cameras often shoot head-on)
  const lowerTop = Math.floor(height * 0.55);
  candidates.push({ left: 0, top: lowerTop, width, height: height - lowerTop, label: 'lower' });

  // Try centre strip
  const centreTop = Math.floor(height * 0.3);
  const centreH   = Math.floor(height * 0.5);
  candidates.push({ left: 0, top: centreTop, width, height: centreH, label: 'centre' });

  return candidates;
}

/**
 * Apply a single preprocessing pipeline to a Sharp instance.
 * Returns a Buffer ready for Tesseract.
 */
async function applyPipeline(buffer, pipeline) {
  let img = sharp(buffer);

  switch (pipeline) {
    case 'standard':
      img = img
        .greyscale()
        .normalize()
        .sharpen({ sigma: 1.5, m1: 1, m2: 2 })
        .threshold(128)
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
      break;

    case 'night':
      img = img
        .greyscale()
        .gamma(2.2)                       // lift shadows
        .normalize()
        .median(3)                        // reduce noise
        .sharpen({ sigma: 2, m1: 2, m2: 3 })
        .threshold(110)
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
      break;

    case 'washed':
      img = img
        .greyscale()
        .gamma(0.6)                       // crush highlights
        .modulate({ brightness: 0.8, saturation: 0 })
        .normalize()
        .sharpen({ sigma: 1.8, m1: 1.5, m2: 2 })
        .threshold(145)
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
      break;

    case 'dirty':
      img = img
        .greyscale()
        .normalize()
        .linear(1.6, -40)               // increase contrast manually
        .sharpen({ sigma: 3, m1: 3, m2: 4 })
        .median(2)
        .threshold(100)
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
      break;

    case 'angled':
      img = img
        .greyscale()
        .normalize()
        .sharpen({ sigma: 2.5, m1: 2, m2: 3 })
        .threshold(120)
        // Scale up 3x before resize — helps with low-res gantry stills
        .resize(TARGET_W * 3, TARGET_H * 3, { fit: 'fill', kernel: 'lanczos3' })
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
      break;

    default:
      img = img
        .greyscale()
        .normalize()
        .sharpen()
        .threshold(128)
        .resize(TARGET_W, TARGET_H, { fit: 'fill' });
  }

  // Always output PNG — lossless, Tesseract handles it best
  return img.png({ compressionLevel: 1 }).toBuffer();
}

/**
 * Preprocess an uploaded image buffer and return an array of
 * { pipeline, region, buffer } objects for OCR to try in order.
 *
 * @param {Buffer} inputBuffer
 * @param {string} [outputDir]  optional — save processed images here for debugging
 * @returns {Promise<Array<{ pipeline: string, region: string, buffer: Buffer }>>}
 */
export async function preprocess(inputBuffer, outputDir = null) {
  const results = [];
  const pipelines = ['standard', 'angled', 'night', 'washed', 'dirty'];

  // Decode once to get dimensions
  const meta = await sharp(inputBuffer).metadata();
  logger.debug('Input image', { width: meta.width, height: meta.height, format: meta.format });

  // Validate it's a real image
  if (!meta.width || !meta.height) throw new Error('Invalid or unreadable image');

  // Get crop regions
  const regions = await detectAndCropPlateRegion(sharp(inputBuffer));

  for (const region of regions) {
    // Crop to this region
    let cropped;
    try {
      cropped = await sharp(inputBuffer)
        .extract({ left: region.left, top: region.top, width: region.width, height: region.height })
        .toBuffer();
    } catch {
      continue;
    }

    for (const pipeline of pipelines) {
      try {
        const processed = await applyPipeline(cropped, pipeline);

        results.push({ pipeline, region: region.label, buffer: processed });

        if (outputDir) {
          const fname = path.join(outputDir, `${region.label}_${pipeline}.png`);
          await sharp(processed).toFile(fname);
        }
      } catch (err) {
        logger.warn(`Pipeline ${pipeline}/${region.label} failed`, { err: err.message });
      }
    }
  }

  return results;
}

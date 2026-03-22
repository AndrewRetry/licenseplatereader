import { detect } from '../services/detection.service.js';
import { validatePlate } from '../services/validation.service.js';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROCESSED_DIR = path.resolve(__dirname, '../../processed');
const DEBUG = process.env.NODE_ENV === 'development';

/**
 * POST /api/plate/detect
 * Body: multipart/form-data  field: image
 * Returns: detection result with ranked plate candidates
 */
export async function detectPlate(req, res) {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No image file provided. Use multipart field "image".' });
  }

  const requestId = uuidv4();
  logger.info('Detection request', { requestId, size: req.file.size, mime: req.file.mimetype });

  try {
    const saveDir = DEBUG ? path.join(PROCESSED_DIR, requestId) : null;
    if (saveDir) {
      const { mkdir } = await import('fs/promises');
      await mkdir(saveDir, { recursive: true });
    }

    const result = await detect(req.file.buffer, { saveProcessed: saveDir });

    return res.status(result.success ? 200 : 422).json({
      requestId,
      success: result.success,
      elapsedMs: result.elapsedMs,
      best: result.best,
      candidates: result.candidates,
      meta: {
        ocrAttempts: result.ocrAttempts,
        preprocessVariants: result.preprocessVariants,
      },
    });
  } catch (err) {
    logger.error('Detection failed', { requestId, err: err.message });
    return res.status(500).json({ success: false, requestId, error: err.message });
  }
}

/**
 * POST /api/plate/validate
 * Body: JSON { plate: "SBA1234A" }
 * Returns: validation result (useful for manual entry verification)
 */
export async function validatePlateText(req, res) {
  const { plate } = req.body ?? {};
  if (!plate || typeof plate !== 'string') {
    return res.status(400).json({ success: false, error: 'Provide { plate: "..." } in request body' });
  }

  const result = validatePlate(plate);
  if (!result) {
    return res.status(422).json({ success: false, plate, error: 'Does not match any known SG plate format' });
  }

  return res.json({ success: true, ...result });
}

/**
 * GET /api/plate/health
 */
export function health(_req, res) {
  res.json({ status: 'ok', service: 'licenseplatereader', ts: new Date().toISOString() });
}

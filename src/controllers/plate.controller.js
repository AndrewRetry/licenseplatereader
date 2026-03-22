import { detect } from '../services/detection.service.js';
import { validatePlate } from '../services/validation.service.js';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname   = path.dirname(fileURLToPath(import.meta.url));
const PROCESSED_DIR = path.resolve(__dirname, '../../processed');
const DEBUG       = process.env.NODE_ENV === 'development';

/**
 * Per-IP in-flight guard.
 *
 * Auto-scan fires every 2s from the webcam client, but each detection
 * request takes 1–10s depending on image complexity. Without a guard,
 * 5–10 requests pile up, all waiting for the same 2-worker pool.
 * The queued requests are stale frames — there is no value processing them.
 *
 * If an IP already has a scan in flight: return 202 Accepted with a
 * "scan_in_progress" flag so the client knows to skip this cycle.
 * We don't 429 because that would trigger error handling on the client.
 */
const inFlight = new Set(); // IP addresses currently being processed

// ── POST /api/plate/detect ─────────────────────────────────────────────

export async function detectPlate(req, res) {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: 'No image file provided. Use multipart field "image".',
    });
  }

  const ip        = req.ip ?? 'unknown';
  const requestId = uuidv4();

  // Drop stale queued frames from the same source
  if (inFlight.has(ip)) {
    logger.debug('Dropping in-flight duplicate', { requestId, ip });
    return res.status(202).json({
      success:          false,
      requestId,
      scan_in_progress: true,
      error:            'A scan is already in progress. Retry after it completes.',
    });
  }

  inFlight.add(ip);
  logger.info('Detection request', { requestId, ip, size: req.file.size, mime: req.file.mimetype });

  try {
    let saveDir = null;
    if (DEBUG) {
      const { mkdir } = await import('fs/promises');
      saveDir = path.join(PROCESSED_DIR, requestId);
      await mkdir(saveDir, { recursive: true });
    }

    const result = await detect(req.file.buffer, { saveProcessed: saveDir });

    return res.status(result.success ? 200 : 422).json({
      requestId,
      success:    result.success,
      elapsedMs:  result.elapsedMs,
      best:       result.best,
      candidates: result.candidates,
      meta: {
        ocrAttempts:       result.ocrAttempts,
        preprocessVariants: result.preprocessVariants,
      },
    });
  } catch (err) {
    logger.error('Detection failed', { requestId, err: err.message });
    return res.status(500).json({ success: false, requestId, error: err.message });
  } finally {
    inFlight.delete(ip);
  }
}

// ── POST /api/plate/validate ───────────────────────────────────────────

export async function validatePlateText(req, res) {
  const { plate } = req.body ?? {};
  if (!plate || typeof plate !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Provide { plate: "..." } in request body',
    });
  }

  const result = validatePlate(plate);
  if (!result) {
    return res.status(422).json({
      success: false, plate, error: 'Does not match any known SG plate format',
    });
  }

  return res.json({ success: true, ...result });
}

// ── GET /api/plate/health ──────────────────────────────────────────────

export function health(_req, res) {
  res.json({
    status:    'ok',
    service:   'licenseplatereader',
    ts:        new Date().toISOString(),
    workers:   parseInt(process.env.WORKER_POOL_SIZE ?? '2', 10),
    inFlight:  inFlight.size,
  });
}
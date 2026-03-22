import { detect } from '../services/detection.service.js';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * POST /api/plate/detect/batch
 * Body: multipart/form-data  field: images (up to 10 files)
 *
 * Returns per-image results in the same order as uploaded.
 * Processes sequentially to avoid overwhelming the worker pool.
 */
export async function detectBatch(req, res) {
  const files = req.files;

  if (!files || files.length === 0) {
    return res.status(400).json({ success: false, error: 'No images provided. Use multipart field "images".' });
  }

  if (files.length > 10) {
    return res.status(400).json({ success: false, error: 'Maximum 10 images per batch request.' });
  }

  const batchId = uuidv4();
  const start = Date.now();
  logger.info('Batch detection request', { batchId, count: files.length });

  const results = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const itemId = `${batchId}-${i}`;

    try {
      const result = await detect(file.buffer);
      results.push({
        index: i,
        filename: file.originalname,
        success: result.success,
        best: result.best,
        candidates: result.candidates,
        elapsedMs: result.elapsedMs,
      });
    } catch (err) {
      logger.warn('Batch item failed', { itemId, err: err.message });
      results.push({
        index: i,
        filename: file.originalname,
        success: false,
        error: err.message,
      });
    }
  }

  const totalMs = Date.now() - start;
  const succeeded = results.filter(r => r.success).length;

  logger.info('Batch complete', { batchId, succeeded, total: files.length, totalMs });

  return res.status(200).json({
    batchId,
    totalMs,
    summary: { total: files.length, succeeded, failed: files.length - succeeded },
    results,
  });
}

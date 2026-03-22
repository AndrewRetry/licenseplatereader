import { Router } from 'express';
import { upload } from '../middleware/upload.js';
import { rateLimit } from '../middleware/rateLimit.js';
import { detectPlate, validatePlateText, health } from '../controllers/plate.controller.js';
import { detectBatch } from '../controllers/batch.controller.js';

const router = Router();

// 30 req/min for single detect, 10 req/min for batch
const detectLimiter = rateLimit({ windowMs: 60_000, max: 30 });
const batchLimiter  = rateLimit({ windowMs: 60_000, max: 10 });

/**
 * @route  GET  /api/plate/health
 */
router.get('/health', health);

/**
 * @route  POST /api/plate/detect
 * @body   multipart/form-data  field: image
 */
router.post('/detect', detectLimiter, upload.single('image'), detectPlate);

/**
 * @route  POST /api/plate/detect/batch
 * @body   multipart/form-data  field: images (up to 10)
 */
router.post('/detect/batch', batchLimiter, upload.array('images', 10), detectBatch);

/**
 * @route  POST /api/plate/validate
 * @body   application/json  { plate: string }
 */
router.post('/validate', validatePlateText);

export default router;

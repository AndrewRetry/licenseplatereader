import { createWorker } from 'tesseract.js';
import { logger } from '../utils/logger.js';

const POOL_SIZE = parseInt(process.env.WORKER_POOL_SIZE ?? '2', 10);
const MIN_CONFIDENCE = parseInt(process.env.MIN_CONFIDENCE ?? '45', 10);

/**
 * Tesseract config tuned for licence plate characters:
 *  - PSM 7  = treat image as a single text line (best for cropped plates)
 *  - PSM 8  = single word
 *  - PSM 13 = raw line (no dictionary, no segmentation)
 * We try all three and take the best.
 */
const PSM_MODES = [7, 8, 13];

/**
 * Whitelist: only uppercase letters and digits appear on SG plates.
 * Feeding this to Tesseract dramatically cuts misreads.
 */
const CHAR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

let pool = [];
let ready = false;

export async function initWorkerPool() {
  logger.info(`Initialising Tesseract worker pool`, { size: POOL_SIZE });

  pool = await Promise.all(
    Array.from({ length: POOL_SIZE }, async () => {
      const worker = await createWorker('eng', 1, {
        logger: m => {
          if (m.status === 'recognizing text') {
            logger.debug('Tesseract progress', { progress: m.progress });
          }
        },
      });

      await worker.setParameters({
        tessedit_char_whitelist: CHAR_WHITELIST,
        tessedit_pageseg_mode: '7',
        // Disable dictionary lookups — plates are not real words
        load_system_dawg: '0',
        load_freq_dawg: '0',
        // Improve number/letter disambiguation
        classify_bln_numeric_mode: '0',
      });

      return { worker, busy: false };
    })
  );

  ready = true;
  logger.info('Worker pool ready');
}

export async function destroyWorkerPool() {
  await Promise.all(pool.map(({ worker }) => worker.terminate()));
  pool = [];
  ready = false;
}

/** Acquire a free worker from pool (simple round-robin with busy flag) */
async function acquireWorker() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const slot = pool.find(w => !w.busy);
    if (slot) { slot.busy = true; return slot; }
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error('Worker pool exhausted — all workers busy for >30s');
}

function releaseWorker(slot) { slot.busy = false; }

/**
 * Run OCR on a single buffer with a specific PSM mode.
 * Returns { text, confidence } or null on failure.
 */
async function recogniseWithPSM(worker, buffer, psm) {
  await worker.setParameters({ tessedit_pageseg_mode: String(psm) });

  const { data } = await worker.recognize(buffer);
  const text = data.text.trim().replace(/\s+/g, ' ');
  const confidence = data.confidence;

  return { text, confidence, psm };
}

/**
 * Run OCR against a set of preprocessed image buffers.
 * Tries every PSM mode on every buffer and returns the best candidate.
 *
 * @param {Array<{ pipeline: string, region: string, buffer: Buffer }>} preprocessed
 * @returns {Promise<Array<{ text: string, confidence: number, pipeline: string, region: string, psm: number }>>}
 */
export async function recognise(preprocessed) {
  if (!ready) throw new Error('Worker pool not initialised');

  const slot = await acquireWorker();
  const candidates = [];

  try {
    for (const { pipeline, region, buffer } of preprocessed) {
      for (const psm of PSM_MODES) {
        try {
          const result = await recogniseWithPSM(slot.worker, buffer, psm);
          if (result.confidence >= MIN_CONFIDENCE && result.text.length >= 3) {
            candidates.push({ ...result, pipeline, region });
            logger.debug('OCR candidate', { ...result, pipeline, region });
          }
        } catch (err) {
          logger.warn('OCR attempt failed', { pipeline, region, psm, err: err.message });
        }
      }
    }
  } finally {
    releaseWorker(slot);
  }

  // Sort by confidence descending
  candidates.sort((a, b) => b.confidence - a.confidence);
  return candidates;
}

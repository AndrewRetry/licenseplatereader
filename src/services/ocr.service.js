import { createWorker } from 'tesseract.js';
import { logger } from '../utils/logger.js';

const POOL_SIZE       = parseInt(process.env.WORKER_POOL_SIZE ?? '2', 10);
const MIN_CONFIDENCE  = parseInt(process.env.MIN_CONFIDENCE   ?? '45', 10);
const EARLY_EXIT_CONF = parseInt(process.env.EARLY_EXIT_CONF  ?? '80', 10);

/**
 * PSM modes per variant.
 * PSM 7  = single text line  — best for correctly-cropped plate strip
 * PSM 8  = single word       — catches tight plates with no spacing
 * PSM 6  = uniform block     — fallback when plate region is loose
 *
 * PSM 13 removed — rarely beats 7/8 on plate images, adds ~33% time.
 */
const PSM_MODES = [7, 8, 6];

const CHAR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ';

/**
 * TESSDATA_PATH must point to a directory containing eng.traineddata.gz
 * Copy from /usr/share/tesseract-ocr/5/tessdata/eng.traineddata, then:
 *   gzip -k eng.traineddata
 * Place the .gz in your project's tessdata/ dir and set env:
 *   TESSDATA_PATH=./tessdata
 */
const TESSDATA_PATH = process.env.TESSDATA_PATH ?? null;

let pool  = [];
let ready = false;

// ── Pool lifecycle ────────────────────────────────────────────────────────

export async function initWorkerPool() {
  logger.info('Initialising Tesseract worker pool', { size: POOL_SIZE, tessdata: TESSDATA_PATH ?? 'CDN' });

  pool = await Promise.all(
    Array.from({ length: POOL_SIZE }, async () => {
      /**
       * INIT-PARAM BUG FIX:
       * load_system_dawg / load_freq_dawg are Tesseract "init-only" parameters.
       * Passing them to worker.setParameters() is silently ignored (tesseract.js v5
       * prints a warning but does not apply them). The English dictionary stays
       * loaded, causing word-substitution noise on plate reads (e.g. "SNL" → "SNI",
       * "3666" → "366 E"). They must be supplied at createWorker time via the
       * `initOptions` / workerOptions pathway — which tesseract.js exposes by
       * including them in the options object itself (they get forwarded to the
       * underlying Tesseract init call before the DAWG data is loaded).
       */
      const worker = await createWorker('eng', 1, {
        ...(TESSDATA_PATH ? { langPath: TESSDATA_PATH } : {}),
        cacheMethod: TESSDATA_PATH ? 'none' : 'write',
        logger: m => {
          if (m.status === 'recognizing text') {
            logger.debug('Tesseract progress', { progress: m.progress });
          }
        },
        // Init-only params — must be here, NOT in setParameters()
        load_system_dawg:          '0',
        load_freq_dawg:            '0',
        load_unambig_dawg:         '0',
        load_punc_dawg:            '0',
        load_number_dawg:          '0',
        load_bigram_dawg:          '0',
        language_model_ngram_on:   '0',
      });

      await worker.setParameters({
        tessedit_char_whitelist:   CHAR_WHITELIST,
        tessedit_pageseg_mode:     '7',
        classify_bln_numeric_mode: '0',
      });

      return { worker, busy: false, currentPsm: 7 };
    })
  );

  ready = true;
  logger.info('Worker pool ready', { size: pool.length });
}

export async function destroyWorkerPool() {
  await Promise.all(pool.map(({ worker }) => worker.terminate()));
  pool  = [];
  ready = false;
}

// ── Worker acquisition ────────────────────────────────────────────────────

async function acquireWorker(timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const slot = pool.find(w => !w.busy);
    if (slot) { slot.busy = true; return slot; }
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error('Worker pool exhausted — all workers busy for >30s');
}

function releaseWorker(slot) { slot.busy = false; }

// ── OCR core ──────────────────────────────────────────────────────────────

async function recogniseWithPSM(slot, buffer, psm) {
  if (slot.currentPsm !== psm) {
    await slot.worker.setParameters({ tessedit_pageseg_mode: String(psm) });
    slot.currentPsm = psm;
  }
  const { data } = await slot.worker.recognize(buffer);
  return {
    text:       data.text.trim().replace(/\s+/g, ' '),
    confidence: data.confidence,
    psm,
  };
}

async function processChunk(variants, earlyExit) {
  const slot       = await acquireWorker();
  const candidates = [];

  try {
    for (const { pipeline, region, buffer } of variants) {
      if (earlyExit.done) break;

      for (const psm of PSM_MODES) {
        if (earlyExit.done) break;
        try {
          const result = await recogniseWithPSM(slot, buffer, psm);

          if (result.confidence >= MIN_CONFIDENCE && result.text.length >= 3) {
            candidates.push({ ...result, pipeline, region });
            logger.debug('OCR candidate', {
              text:       result.text,
              conf:       result.confidence,
              pipeline,
              region,
              psm,
            });

            if (result.confidence >= EARLY_EXIT_CONF) {
              earlyExit.done = true;
              logger.debug('Early exit', { conf: result.confidence, text: result.text });
              break;
            }
          }
        } catch (err) {
          logger.warn('OCR attempt failed', { pipeline, region, psm, err: err.message });
        }
      }
    }
  } finally {
    releaseWorker(slot);
  }

  return candidates;
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Run OCR across all preprocessed variants using all pool workers in parallel.
 *
 * PARALLELISM FIX: original code acquired one worker and ran all variants
 * sequentially, leaving the rest of the pool idle. Now variants are split into
 * pool.length chunks and processed concurrently.
 *
 * EARLY EXIT: a shared flag stops all workers once a high-confidence result
 * is found — on a clear plate this fires after 1–2 OCR calls (~200–400ms).
 */
export async function recognise(preprocessed) {
  if (!ready)                  throw new Error('Worker pool not initialised');
  if (!preprocessed.length)    return [];

  const earlyExit  = { done: false };
  const chunkCount = Math.min(pool.length, preprocessed.length);
  const chunkSize  = Math.ceil(preprocessed.length / chunkCount);

  const chunks = [];
  for (let i = 0; i < preprocessed.length; i += chunkSize) {
    chunks.push(preprocessed.slice(i, i + chunkSize));
  }

  const all = (await Promise.all(chunks.map(c => processChunk(c, earlyExit)))).flat();
  all.sort((a, b) => b.confidence - a.confidence);
  return all;
}
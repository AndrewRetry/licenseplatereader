import { preprocess } from './preprocessing.service.js';
import { recognise } from './ocr.service.js';
import { validatePlate, normalise } from './validation.service.js';
import { logger } from '../utils/logger.js';

/**
 * Main detection pipeline.
 *
 * Steps:
 *  1. Preprocess image into multiple variants (pipelines × regions)
 *  2. Run OCR on all variants, collect candidates with confidence scores
 *  3. Validate each candidate against SG plate patterns
 *  4. Score candidates: OCR confidence × format match × checksum bonus
 *  5. Return ranked results
 *
 * @param {Buffer} imageBuffer
 * @param {{ debug?: boolean, saveProcessed?: string }} [opts]
 * @returns {Promise<DetectionResult>}
 */
export async function detect(imageBuffer, opts = {}) {
  const start = Date.now();

  // Step 1 — preprocess
  let preprocessed;
  try {
    preprocessed = await preprocess(imageBuffer, opts.saveProcessed ?? null);
    logger.debug('Preprocessing done', { variants: preprocessed.length });
  } catch (err) {
    throw new Error(`Preprocessing failed: ${err.message}`);
  }

  if (preprocessed.length === 0) throw new Error('No processable image variants produced');

  // Step 2 — OCR
  const ocrCandidates = await recognise(preprocessed);
  logger.debug('OCR candidates', { count: ocrCandidates.length });

  // Step 3 & 4 — Validate + score
  const validated = [];
  const seen = new Set();

  for (const candidate of ocrCandidates) {
    const validation = validatePlate(candidate.text);
    if (!validation) continue;

    // Deduplicate by normalised plate string
    if (seen.has(validation.plate)) {
      // If we saw this plate before, pick the higher-scored entry
      const existing = validated.find(v => v.plate === validation.plate);
      if (existing && candidate.confidence > existing.ocrConfidence) {
        existing.ocrConfidence = candidate.confidence;
        existing.pipeline = candidate.pipeline;
        existing.psm = candidate.psm;
        existing.region = candidate.region;
        existing.score = computeScore(candidate.confidence, validation);
      }
      continue;
    }

    seen.add(validation.plate);

    validated.push({
      ...validation,
      ocrConfidence: candidate.confidence,
      pipeline: candidate.pipeline,
      region: candidate.region,
      psm: candidate.psm,
      rawOcrText: candidate.text,
      score: computeScore(candidate.confidence, validation),
    });
  }

  // Sort by composite score
  validated.sort((a, b) => b.score - a.score);

  const elapsed = Date.now() - start;

  logger.info('Detection complete', {
    found: validated.length,
    top: validated[0]?.plate ?? null,
    topScore: validated[0]?.score ?? null,
    elapsedMs: elapsed,
  });

  return {
    success: validated.length > 0,
    elapsedMs: elapsed,
    candidates: validated,
    best: validated[0] ?? null,
    ocrAttempts: ocrCandidates.length,
    preprocessVariants: preprocessed.length,
  };
}

/**
 * Composite score:
 *  - Base: OCR confidence (0–100)
 *  - +15 bonus for valid checksum
 *  - -20 penalty for invalid checksum
 *  - +5 bonus for known high-frequency formats
 */
function computeScore(ocrConf, validation) {
  let score = ocrConf;

  if (validation.checksumValid === true) score += 15;
  if (validation.checksumValid === false) score -= 20;
  if (['private_car', 'taxi', 'private_hire'].includes(validation.format)) score += 5;

  return Math.max(0, Math.min(120, score));
}

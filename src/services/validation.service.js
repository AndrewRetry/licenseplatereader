/**
 * Singapore License Plate Validation Service
 *
 * SG plate formats:
 *  Standard private car  : [A-Z]{1,3} [0-9]{1,4} [A-Z]   e.g. SBA 1234 A
 *  Taxi                  : SH[A-Z]? [0-9]{4} [A-Z]       e.g. SHA 1234 B
 *  Motorcycle            : F[A-Z] [0-9]{1,4} [A-Z]?
 *  Government            : QX [0-9]{4}
 *  Private hire (PHV)    : SX [0-9]{4} [A-Z]
 *  Diplomatic / consular : D [0-9]{4} [A-Z]?
 *  Classic red plate     : [A-Z]{1,3} [0-9]{1,4} (no checksum)
 *
 * All patterns are anchored and case-insensitive after normalisation.
 */

// Checksum letter lookup table (Mod 19)
const CHECKSUM_LETTERS = 'AZYXUTSRPMLKJHGEDCB';

const PATTERNS = [
  // Standard: SBA1234A  or  S1234A  or  SA1234A
  /^([A-Z]{1,3})(\d{1,4})([A-Z])$/,
  // Motorcycle: FBA123 or FB123A
  /^(F[A-Z])(\d{1,4})([A-Z]?)$/,
  // Government: QX1234
  /^(QX)(\d{4})$/,
  // Diplomatic: D1234A
  /^(D)(\d{4})([A-Z]?)$/,
];

const WEIGHTS = [9, 4, 5, 4, 3, 2];
const PREFIX_PADDING = { 1: '  ', 2: ' ', 3: '' };

/**
 * Compute the expected checksum letter for a SG plate.
 * @param {string} prefix  e.g. "SBA"
 * @param {string} digits  e.g. "1234"
 * @returns {string} single uppercase letter
 */
export function computeChecksum(prefix, digits) {
  const padded = (PREFIX_PADDING[prefix.length] + prefix).slice(-3);
  // Weights cover 3 prefix positions + first 3 of the 4-digit zero-padded number = 6 total
  const nums = [
    ...padded.split('').map(c => (c === ' ' ? 0 : c.charCodeAt(0) - 64)),
    ...digits.padStart(4, '0').slice(0, 3).split('').map(Number),
  ];
  const sum = nums.reduce((acc, n, i) => acc + n * WEIGHTS[i], 0);
  return CHECKSUM_LETTERS[sum % 19];
}

/**
 * Normalise raw OCR text into a candidate plate string.
 * Handles common OCR misreads: 0↔O, 1↔I, 8↔B, 5↔S etc.
 */
export function normalise(raw) {
  return raw
    .toUpperCase()
    .replace(/\s+/g, '')           // strip all whitespace
    .replace(/[^A-Z0-9]/g, '')     // strip non-alphanumeric
    .replace(/O(?=\d)/g, '0')      // O → 0 when adjacent to digit
    .replace(/(?<=\d)O/g, '0')
    .replace(/I(?=\d)/g, '1')
    .replace(/(?<=\d)I/g, '1')
    .replace(/(?<=[A-Z])8(?=[A-Z])/g, 'B');  // 8 → B between letters
}

/**
 * Validate a normalised plate string against all known SG patterns.
 * Returns an enriched result object or null if invalid.
 *
 * @param {string} raw  raw OCR text
 * @returns {{ plate: string, prefix: string, digits: string, checksum: string|null,
 *             checksumValid: boolean|null, format: string, confidence: 'high'|'medium'|'low' } | null}
 */
export function validatePlate(raw) {
  if (!raw || raw.length < 3) return null;

  const candidate = normalise(raw);

  for (const pattern of PATTERNS) {
    const match = candidate.match(pattern);
    if (!match) continue;

    const [, prefix, digits, checksum = null] = match;

    let checksumValid = null;
    if (checksum) {
      const expected = computeChecksum(prefix, digits);
      checksumValid = expected === checksum;
    }

    const format = resolveFormat(prefix);
    const confidence = checksumValid === true ? 'high'
      : checksumValid === false ? 'low'
      : 'medium';

    return {
      plate: candidate,
      formatted: formatPlate(prefix, digits, checksum),
      prefix,
      digits,
      checksum,
      checksumValid,
      format,
      confidence,
    };
  }

  return null;
}

function resolveFormat(prefix) {
  if (prefix === 'QX') return 'government';
  if (prefix === 'D')  return 'diplomatic';
  if (prefix.startsWith('F')) return 'motorcycle';
  if (prefix.startsWith('SH')) return 'taxi';
  if (prefix.startsWith('SX')) return 'private_hire';
  return 'private_car';
}

function formatPlate(prefix, digits, checksum) {
  return checksum
    ? `${prefix} ${digits.replace(/^0+/, '') || '0'} ${checksum}`
    : `${prefix} ${digits.replace(/^0+/, '') || '0'}`;
}

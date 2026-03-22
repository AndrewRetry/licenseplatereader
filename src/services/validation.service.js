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
  const nums = [
    ...padded.split('').map(c => (c === ' ' ? 0 : c.charCodeAt(0) - 64)),
    ...digits.padStart(4, '0').slice(0, 3).split('').map(Number),
  ];
  const sum = nums.reduce((acc, n, i) => acc + n * WEIGHTS[i], 0);
  return CHECKSUM_LETTERS[sum % 19];
}

/**
 * Normalise raw OCR text into a candidate plate string.
 *
 * Key fix: O→0 and I→1 substitutions are scoped exclusively to the digit
 * block, determined by splitting the cleaned string into
 * (letter-prefix)(digits)(optional-trailing-letter).
 *
 * Previous approach used lookahead/lookbehind on the raw string which caused:
 *   1. Prefix letters adjacent to digits being replaced (SBO1234A → SB01234A,
 *      giving 5-digit block that fails every pattern)
 *   2. The checksum letter being replaced (SBA1234O → SBA12340, trailing
 *      digit breaks the [A-Z] requirement)
 */
export function normalise(raw) {
  const clean = raw
    .toUpperCase()
    .replace(/\s+/g, '')
    .replace(/[^A-Z0-9]/g, '');

  const m = clean.match(/^([A-Z]*)(\d+)([A-Z]?)$/);
  if (!m) return clean;

  const [, prefix, digits, checksum] = m;

  // O/I → digit only inside the digit block
  const fixedDigits = digits
    .replace(/O/g, '0')
    .replace(/I/g, '1');

  // 8 → B only inside the prefix (between letters)
  const fixedPrefix = prefix.replace(/(?<=[A-Z])8(?=[A-Z]|$)/g, 'B');

  return fixedPrefix + fixedDigits + checksum;
}

/**
 * Validate a normalised plate string against all known SG patterns.
 * Returns an enriched result object or null if invalid.
 *
 * @param {string} raw  raw OCR text
 * @returns {{ plate, prefix, digits, checksum, checksumValid, format, confidence } | null}
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
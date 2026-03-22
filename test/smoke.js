/**
 * Smoke test — validates core logic without spinning up the HTTP server.
 * Run: node test/smoke.js
 */

import { validatePlate, computeChecksum, normalise } from '../src/services/validation.service.js';

let pass = 0, fail = 0;

function assert(label, got, expected) {
  const ok = JSON.stringify(got) === JSON.stringify(expected);
  console.log(`  ${ok ? '✓' : '✗'} ${label}${ok ? '' : `\n      got:      ${JSON.stringify(got)}\n      expected: ${JSON.stringify(expected)}`}`);
  ok ? pass++ : fail++;
}

// ── normalise ─────────────────────────────────────────────────────────────
console.log('\nnormalise()');
assert('strips spaces',       normalise('S B A 1 2 3 4'), 'SBA1234');
assert('O→0 before digit',    normalise('SBO1234A'),      'SB01234A');
assert('I→1 before digit',    normalise('SBI234A'),       'SB1234A');
assert('strips special chars', normalise('SBA-1234!A'),   'SBA1234A');

// ── computeChecksum ───────────────────────────────────────────────────────
console.log('\ncomputeChecksum()');
// Round-trip: compute then verify
const cs1 = computeChecksum('SBA', '1234');
assert('returns a letter', typeof cs1 === 'string' && /^[A-Z]$/.test(cs1), true);
const cs2 = computeChecksum('E', '1');
assert('short prefix works', typeof cs2 === 'string' && /^[A-Z]$/.test(cs2), true);

// ── validatePlate ─────────────────────────────────────────────────────────
console.log('\nvalidatePlate()');

// Build plates with correct checksums
const p1 = `SBA1234${computeChecksum('SBA', '1234')}`;
const r1 = validatePlate(p1);
assert('private car — valid checksum → high', r1?.confidence, 'high');
assert('private car — checksumValid true',    r1?.checksumValid, true);
assert('private car — format',                r1?.format, 'private_car');

const r2 = validatePlate('QX1234');
assert('government — no checksum → medium', r2?.confidence, 'medium');
assert('government — format',              r2?.format, 'government');

const r3 = validatePlate('SHA5678X');
assert('taxi — format',  r3?.format, 'taxi');

assert('invalid string → null',  validatePlate('XYZ'), null);
assert('too short → null',       validatePlate('AB'),  null);
assert('numbers only → null',    validatePlate('1234'), null);

// ── spaced input normalised ───────────────────────────────────────────────
console.log('\nspaced / dirty input');
const r4 = validatePlate('S B A 1 2 3 4 A');
assert('spaced plate parsed', r4?.plate, 'SBA1234A');
assert('O→0 coercion', validatePlate('SBAO234A')?.digits, '0234');

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${pass + fail} tests — ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);

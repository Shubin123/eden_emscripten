// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs for the WebAssembly backend's regression and speed tests: loops with several
// blocks and forward and backward edges, calls, a block that branches to itself, and a compare
// and branch for every condition. Each runs as one input of dynarmic_test_reader.
//
//   node wasm_jit_programs.mjs <dir with dynarmic_test_reader.js> [--bench <variant dir> ...] [--rounds N]
// Without --bench: runs each program through the IR interpreter and the Wasm JIT and checks that
// the final guest state matches (quick; a few seconds). With --bench: reports guest instructions
// per second of each build's test reader, interleaved so machine load hits them alike; the first
// directory is the reference. Takes the median of --rounds (default 5).

import { spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';

const { values: args, positionals } = parseArgs({
    allowPositionals: true,
    options: {
        bench: { type: 'boolean', default: false },
        rounds: { type: 'string', default: '5' },
        ticks: { type: 'string', default: '3000' },
    },
});

// Assembled with `clang -arch arm64`; the comments are the source.
const PROGRAMS = {
    // Four-way branching loop: forward edges that skip blocks, a back edge, flag-setting compares.
    loop: [0x91000421, 0x92400422, 0xb4000062, 0xca010063, 0x14000002, 0x8b010463, 0xf27e003f, 0x540000c0,
        0xcb030084, 0xeb05009f, 0x5400004b, 0x910004c6, 0xaa0400e7, 0x8b0400a5, 0xca0700a8, 0x93c83509,
        0x8b09014a, 0x17ffffef],
    // bl/ret around a small leaf function.
    call: [0x91000421, 0x92400824, 0x94000006, 0x8b000063, 0xca0300a5, 0xb4000044, 0x8b0500c6, 0x17fffff9,
        0xca030020, 0x91001c00, 0x93c00c00, 0xd65f03c0],
    // A block that loops on itself (subs; b.ne) inside an outer loop.
    self: [0x91000421, 0xca010042, 0x8b020863, 0xcac31ca5, 0xf1000484, 0x54ffff61, 0x17fffffa],
    // Lane inserts (ins .b/.h/.d), trn1/trn2, zip1/zip2 and ext on 8- and 16-byte vectors, shifts
    // by immediates up to the lane width, uxtl/sxtl, dup into 64 bits, vector ucvtf/scvtf, and
    // scalar ucvtf/scvtf/fcvt{zu,zs,ms,ps,ns,as} with and without fixed-point bits, also on raw
    // bit patterns (NaN, infinity, denormals); everything folds into x3-x14 and the vectors.
    simd: [0x91278c21, 0xcac13442, 0x8b011442, 0x4e040c20, 0x4e181c40, 0x4e0e1c40, 0x4e131c20, 0x4e1a1c20,
        0x4ea01c01, 0x4e081c41, 0x4e051c41, 0x4e812802, 0x4e406823, 0x4e012804, 0x4ec16805, 0x0e016806,
        0x4e813807, 0x4e007828, 0x0e013809, 0x4ec1780a, 0x4e40382b, 0x6e01280c, 0x2e01180d, 0x6e00782e,
        0x1e23004f, 0x9e620050, 0x1e03e451, 0x1e02f432, 0x9e43b033, 0x9e220054, 0x1e270055, 0x9e670056,
        0x1e3901e3, 0x9e780204, 0x1e19f225, 0x1e300246, 0x1e280227, 0x9e2001e8, 0x1e240249, 0x1e3902aa,
        0x9e3002ab, 0x1e6802cc, 0x1e58decd, 0x4f275417, 0x6f1d0438, 0x6f200419, 0x4f20043a, 0x4f0e041b,
        0x4f7f543c, 0x2f10a41d, 0x0f08a43e, 0x2f20a43f, 0x6e371c42, 0x6e381c63, 0x6e391c84, 0x6e3a1ca5,
        0x6e3b1cc6, 0x6e3c1ce7, 0x6e3d1d08, 0x6e3e1d29, 0x6e3f1d4a, 0x0e040c57, 0x0e020c38, 0x0e010c59,
        0x6e21d81a, 0x6f3be43b, 0x4f3de41c, 0x4e21d83d, 0x4eb787de, 0x4eba87ff, 0x6e3b1fbd, 0x6e381f9c,
        0x6e391d6b, 0xca040063, 0xca0600a5, 0xca0800e7, 0xca0a0129, 0xca0c016b, 0x8b0d01ce, 0x4e083c4f,
        0x4e183c70, 0x4e083d91, 0x4e183db2, 0x9e6601f3, 0x9e660234, 0x9e660275, 0xca0f01ce, 0xcad00dce,
        0xcad11dce, 0xca1201ce, 0x8b1301ce, 0x8b1401ce, 0x8b1501ce, 0x17ffffa3],
    // Scalar and vector fmax/fmin/fmaxnm/fminnm, fcmgt/fcmge/fcmeq and faddp on singles and
    // doubles, fcvt from single to half, ushl by signed per-lane amounts, and uzp1/uzp2/zip2 on 8-
    // and 16-byte vectors. Operands mix random values with rotating NaNs, signalling NaNs,
    // denormals, signed zeros, infinity and the half range and rounding boundaries.
    simdfp: [0x52aff814, 0x72800034, 0x52aff015, 0x72800035, 0x52802476, 0x52b00017, 0x4e041e90, 0x4e0c1eb0,
        0x4e141ed0, 0x4e1c1ef0, 0x52aff014, 0x52a8eff5, 0x729e0015, 0x52a70ff6, 0x729e0016, 0x52a71017,
        0x72820017, 0x4e041e91, 0x4e0c1eb1, 0x4e141ed1, 0x4e1c1ef1, 0x52a8eff4, 0x729dfff4, 0x52a02015,
        0x52b00016, 0x72802476, 0x4e041e92, 0x4e0c1eb2, 0x4e141ed2, 0x4e1c1ff2, 0xd2efff14, 0xf2800034,
        0xd2f00015, 0xd2802476, 0xd2effe17, 0xf2800037, 0x4e081e93, 0x4e181eb3, 0x4e081ed4, 0x4e181ef4,
        0xd2980818, 0xf2b7e838, 0xf2d827f8, 0xf2e02018, 0xd29c041a, 0xf2bbe43a, 0xf2dc23fa, 0xf2ffe01a,
        0x91278c21, 0xcac13442, 0x8b011442, 0x6e102210, 0x6e116231, 0x6e122252, 0x6e134273, 0x4e040c40,
        0x4e181c20, 0x4e161c20, 0x6f34e401, 0x4f30e402, 0x4e903823, 0x4e827a24, 0x4e822a45, 0x4e24f466,
        0x4ea5f487, 0x4e25c468, 0x4eb0c4a9, 0x6ea4e46a, 0x6e23e4ab, 0x4e25e60c, 0x6e24d46d, 0x2e22d4ae,
        0x1e30486f, 0x1e257a35, 0x1e246a56, 0x1e305857, 0x6e261f18, 0x6e271f39, 0x6e281f5a, 0x6e291f7b,
        0x4eaa879c, 0x6eab879c, 0x6e2c1f9c, 0x6e2d1fbd, 0x6e2e1fde, 0x6e1f23ff, 0x6e2f1fff, 0x6e1f23ff,
        0x6e351fff, 0x6e1f23ff, 0x6e361fff, 0x6e1f23ff, 0x6e371fff, 0x6f6ce406, 0x4e74f667, 0x4ef3c4c8,
        0x4e66c689, 0x6ef3e4ca, 0x6e66e68b, 0x4e66e4cc, 0x6e73d4cd, 0x1e665a6e, 0x1e666a8f, 0x6e271f18,
        0x6e281f39, 0x6e291f5a, 0x4eea877b, 0x6eeb877b, 0x6e2c1f9c, 0x6e2d1fbd, 0x6e2e1fde, 0x6e1e43de,
        0x6e2f1fde, 0x1e23c046, 0x1e23c027, 0x1e23c208, 0x1e23c229, 0x1e23c24a, 0x5e0c044b, 0x1e23c16b,
        0x5e1c044c, 0x1e23c18c, 0x0e023cc3, 0x0e023ce4, 0x0e023d05, 0x0e023d26, 0x0e023d47, 0x0e023d68,
        0x0e023d89, 0x8b0301ce, 0xcac441ce, 0x8b0501ce, 0xcac681ce, 0x8b0701ce, 0xcac8c1ce, 0x8b0901ce,
        0x93ce1dce, 0x4f01e7f5, 0x4f01e416, 0x4e351c15, 0x6e3686b5, 0x6eb54426, 0x4f03e7f6, 0x4f02e417,
        0x4e361c16, 0x6e3786d6, 0x6ef64427, 0x6ea04448, 0x93d82318, 0x93d86319, 0x4e081f15, 0x4e181f35,
        0x6ef54449, 0x93da235a, 0x4e040f56, 0x93da435b, 0x4e0c1f76, 0x93daa35b, 0x4e141f76, 0x6eb6442a,
        0x6e291f7b, 0x6e2a1f9c, 0x6e261f18, 0x6e271f39, 0x6e281f5a, 0x4e811806, 0x4e405827, 0x4e021808,
        0x4ec15809, 0x0e80182a, 0x0e42580b, 0x0e01584c, 0x0e81780d, 0x0e42782e, 0x0e00784f, 0x6e261f7b,
        0x6e271f9c, 0x6e281fbd, 0x6e291fde, 0x4eaa87ff, 0x6e2b1f18, 0x4eac8739, 0x6e2d1f5a, 0x4eae877b,
        0x6e2f1f9c, 0x6e31d646, 0x4e30f647, 0x4eb2c608, 0x6eb0e649, 0x6e261fbd, 0x6e271fde, 0x6e281fff,
        0x4ea98718, 0x17ffff6f],
};
// FPCR values each program runs under (default: 0). Flush-to-zero, default NaN, alternative half
// precision and every rounding mode change which float operations the JIT can emit natively.
const PROGRAM_FPCRS = {
    simd: [0, 0x01000000, 0x00400000, 0x00800000, 0x00C00000, 0x01C00000],
    simdfp: [0, 0x01000000, 0x02000000, 0x03000000, 0x04000000, 0x00400000, 0x00C00000],
};

const hex = word => (word >>> 0).toString(16).padStart(8, '0');
const START_REGS = Array(31).fill('0');

function input(words, regs = START_REGS, fpcr = 0) {
    return [`instructions: ${words.map(hex).join(' ')}`, `initial_regs: ${regs.join(' ')}`,
        `initial_vecs: ${Array(32).fill('0:0').join(' ')}`, 'initial_sp: 10000', 'initial_pstate: 0',
        `initial_fpcr: ${fpcr.toString(16)}`, ''].join('\n');
}

function run(reader, env, stdin) {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [reader, 'a64'], { env: { ...process.env, ...env }, stdio: ['pipe', 'pipe', 'pipe'] });
        let out = '';
        let err = '';
        child.stdout.on('data', chunk => { out += chunk; });
        child.stderr.on('data', chunk => { err += chunk; });
        child.on('error', reject);
        child.on('close', code => (code === 0 ? resolve({ out, err }) : reject(new Error(`exit ${code}: ${err}`))));
        child.stdin.end(stdin);
    });
}

async function pool(tasks, limit = os.availableParallelism()) {
    const results = new Array(tasks.length);
    let next = 0;
    await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, async () => {
        while (next < tasks.length) {
            const i = next++;
            results[i] = await tasks[i]();
        }
    }));
    return results;
}

const finalState = out => out.split('\n').filter(line => /^final_(regs|vecs|pstate|sp|pc)/.test(line)).join('\n');

if (args.bench) {
    const readers = positionals.map(dir => path.join(path.resolve(dir), 'dynarmic_test_reader.js'));
    const rounds = Number(args.rounds);
    const env = { EDEN_BENCH_TICKS: '100000000', EDEN_JIT_THRESHOLD: '1' };
    const median = values => [...values].sort((a, b) => a - b)[values.length >> 1];
    for (const [name, words] of Object.entries(PROGRAMS)) {
        const rates = readers.map(() => []);
        for (let round = 0; round < rounds; ++round) {
            for (const [i, reader] of readers.entries()) {
                const { err } = await run(reader, env, input(words));
                rates[i].push(Number(err.match(/bench: ([0-9.]+)/)[1]));
            }
        }
        const base = median(rates[0]);
        console.log(readers.map((reader, i) => `${name} ${path.basename(path.dirname(reader))}: ${median(rates[i]).toFixed(0)} M/s ` +
            `(${(median(rates[i]) / base).toFixed(2)}x)`).join('   '));
    }
    process.exit(0);
}

const reader = path.join(path.resolve(positionals[0] ?? 'bin'), 'dynarmic_test_reader.js');
let failures = 0;
const INTERP = { EDEN_INTERP_JIT: '0', EDEN_BENCH_TICKS: args.ticks };
const JIT = { EDEN_JIT_THRESHOLD: '1', EDEN_BENCH_TICKS: args.ticks };

// Whole programs: the same guest state after the same number of instructions.
const programResults = await pool(Object.entries(PROGRAMS).flatMap(([name, words]) => (PROGRAM_FPCRS[name] ?? [0]).map(fpcr => async () => {
    const stdin = input(words, START_REGS, fpcr);
    const [interp, jit] = await Promise.all([run(reader, INTERP, stdin), run(reader, JIT, stdin)]);
    const label = fpcr ? `${name} (fpcr ${fpcr.toString(16)})` : name;
    return { name: label, same: finalState(interp.out) === finalState(jit.out), interp: finalState(interp.out), jit: finalState(jit.out) };
})));
for (const r of programResults) {
    if (!r.same) {
        failures++;
        console.log(`FAIL ${r.name}:\n--- interpreter\n${r.interp}\n--- jit\n${r.jit}`);
    }
}
console.log(`programs: ${programResults.length - programResults.filter(r => !r.same).length}/${programResults.length} match`);

// CMP then B.<cond> for every condition, 64 and 32 bit, over edge operand pairs, against the
// architectural result. One straight-line program per condition and width: each pair loads x0 and
// x1, compares, and a B.<cond> that is not taken sets that pair's bit in x9 or x10 (ORR with a
// single-bit immediate). A long chain of conditional blocks, so this also covers forward branches.
const CONDS = ['eq', 'ne', 'cs', 'cc', 'mi', 'pl', 'vs', 'vc', 'hi', 'ls', 'ge', 'lt', 'gt', 'le'];
const M64 = (1n << 64n) - 1n;
const VALUES = [0n, 1n, 0x7fffffffn, 0x80000000n, 0xffffffffn, 0x100000000n, 0x7fffffffffffffffn, 0x8000000000000000n, M64];

function taken(cond, a, b, width) {
    const mask = width === 32 ? 0xffffffffn : M64;
    a &= mask;
    b &= mask;
    const result = (a - b) & mask;
    const sign = x => (x >> BigInt(width - 1)) & 1n;
    const n = sign(result);
    const z = result === 0n ? 1n : 0n;
    const c = a >= b ? 1n : 0n;
    const v = sign(a) !== sign(b) && sign(result) !== sign(a) ? 1n : 0n;
    return {
        eq: z, ne: 1n - z, cs: c, cc: 1n - c, mi: n, pl: 1n - n, vs: v, vc: 1n - v,
        hi: c & (1n - z), ls: 1n - (c & (1n - z)), ge: n === v ? 1n : 0n, lt: n !== v ? 1n : 0n,
        gt: z === 0n && n === v ? 1n : 0n, le: z === 1n || n !== v ? 1n : 0n,
    }[cond];
}

/// movz/movk sequence loading `value` into x<reg>.
function load(reg, value) {
    const words = [(0xD2800000 | (Number(value & 0xffffn) << 5) | reg) >>> 0];
    for (let half = 1; half < 4; ++half) {
        const part = Number((value >> BigInt(16 * half)) & 0xffffn);
        if (part) words.push((0xF2800000 | (half << 21) | (part << 5) | reg) >>> 0);
    }
    return words;
}

const PAIRS = VALUES.flatMap(a => VALUES.map(b => [a, b]));
const x = (out, n) => BigInt('0x' + (out.match(/^final_regs:(.*)$/m)?.[1].trim().split(/\s+/)[n] ?? '0'));
let condBad = 0;
const condResults = await pool(CONDS.flatMap((cond, index) => [64, 32].map(width => async () => {
    const words = [];
    let expected = [0n, 0n];
    PAIRS.forEach(([a, b], pair) => {
        words.push(...load(0, a), ...load(1, b), width === 64 ? 0xEB01001F : 0x6B01001F, (0x54000000 | (2 << 5) | index) >>> 0);
        // Not taken: fall through to ORR acc, acc, #(1 << bit).
        const acc = pair >= 64 ? 10 : 9;
        const bit = pair % 64;
        words.push((0xB2400000 | (((64 - bit) & 63) << 16) | (acc << 5) | acc) >>> 0);
        if (!taken(cond, a, b, width)) expected[pair >= 64 ? 1 : 0] |= 1n << BigInt(bit);
    });
    words.push(0x14000000);  // b .
    const stdin = input(words);
    const env = { EDEN_BENCH_TICKS: String(words.length) };
    const [interp, jit] = await Promise.all([run(reader, { ...env, EDEN_INTERP_JIT: '0' }, stdin), run(reader, { ...env, EDEN_JIT_THRESHOLD: '1' }, stdin)]);
    const got = out => [x(out, 9), x(out, 10)];
    const ok = out => got(out).every((v, i) => v === expected[i]);
    return { cond, width, ok: ok(interp.out) && ok(jit.out), interp: got(interp.out), jit: got(jit.out), expected };
})));
for (const r of condResults.filter(r => !r.ok)) {
    condBad++;
    console.log(`FAIL ${r.width}-bit ${r.cond}: expected ${r.expected.map(v => v.toString(16))}, interpreter ${r.interp.map(v => v.toString(16))}, jit ${r.jit.map(v => v.toString(16))}`);
}
failures += condBad ? 1 : 0;
console.log(`compare and branch: ${condResults.length - condBad}/${condResults.length} condition and width programs match (${PAIRS.length} operand pairs each)`);
process.exit(failures ? 1 : 0);

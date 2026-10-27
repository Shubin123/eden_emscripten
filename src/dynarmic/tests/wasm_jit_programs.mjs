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
};

const hex = word => (word >>> 0).toString(16).padStart(8, '0');
const START_REGS = Array(31).fill('0');

function input(words, regs = START_REGS) {
    return [`instructions: ${words.map(hex).join(' ')}`, `initial_regs: ${regs.join(' ')}`,
        `initial_vecs: ${Array(32).fill('0:0').join(' ')}`, 'initial_sp: 10000', 'initial_pstate: 0',
        'initial_fpcr: 0', ''].join('\n');
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

const finalState = out => out.split('\n').filter(line => /^final_(regs|pstate|sp|pc)/.test(line)).join('\n');

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
const programResults = await pool(Object.entries(PROGRAMS).map(([name, words]) => async () => {
    const [interp, jit] = await Promise.all([run(reader, INTERP, input(words)), run(reader, JIT, input(words))]);
    return { name, same: finalState(interp.out) === finalState(jit.out), interp: finalState(interp.out), jit: finalState(jit.out) };
}));
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

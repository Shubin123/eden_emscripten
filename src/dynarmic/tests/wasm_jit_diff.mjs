// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Differential test for the WebAssembly backend: runs the same A64 code through the IR
// interpreter (EDEN_INTERP_JIT=0) and through the Wasm JIT (EDEN_JIT_THRESHOLD=1) and compares
// the final guest state. It only depends on IR semantics, so it gives the same answer on any
// host that can run Node.js.
//
//   node wasm_jit_diff.mjs <dir with dynarmic_test_reader.js and dynarmic_test_generator.js>
//       [--seeds N] [--iterations N]
// wasm_jit_programs.mjs adds multi-block loops, calls and every condition code, in seconds, and
// measures guest instructions per second for comparing builds.

import { spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';

const { values: args, positionals } = parseArgs({
    allowPositionals: true,
    options: {
        seeds: { type: 'string', default: '8' },
        iterations: { type: 'string', default: '500' },
    },
});
const binDir = path.resolve(positionals[0] ?? 'bin');
const INTERP = { EDEN_INTERP_JIT: '0' };
const JIT = { EDEN_JIT_THRESHOLD: '1' };

function run(script, argv, env, input = '') {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [path.join(binDir, script), ...argv], {
            env: { ...process.env, ...env },
            stdio: ['pipe', 'pipe', 'inherit'],
        });
        let out = '';
        child.stdout.on('data', chunk => { out += chunk; });
        child.on('error', reject);
        child.on('close', code => (code === 0 ? resolve(out) : reject(new Error(`${script} exited with ${code}`))));
        child.stdin.end(input);
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

// Word-at-a-time strlen, the shape libc uses. Both rets are `b .` so the test stops there.
//   mov x8, x0; tst x0, #7; b.eq 2f; ldrb w8, [x0]; cbz w8, 4f; add x8, x0, #1
//   1: tst x8, #7; b.eq 2f; ldrb w9, [x8], #1; cbnz w9, 1b; sub x8, x8, #1; b 3f
//   2: mov x9, #0xfefefefefefefefe; movk x9, #0xfeff; sub x8, x8, #8
//   5: ldr x10, [x8, #8]!; add x11, x10, x9; bic x11, x11, x10; tst x11, #0x8080808080808080
//      b.eq 5b; tst w10, #0xff; b.eq 3f
//   6: ldrb w9, [x8, #1]!; cbnz w9, 6b
//   3: sub x0, x8, x0; b .
//   4: mov x0, xzr; b .
// The final `tst w10` once read stale carry/overflow locals left by other blocks of the region,
// corrupting Z (games then asked sm for "pct" instead of "pctl:s").
const STRLEN = [
    0xaa0003e8, 0xf240081f, 0x54000140, 0x39400008, 0x340002c8, 0x91000408, 0xf240091f,
    0x540000a0, 0x38401509, 0x35ffffa9, 0xd1000508, 0x1400000d, 0xb207dbe9, 0xf29fdfe9,
    0xd1002108, 0xf8408d0a, 0x8b09014b, 0x8a2a016b, 0xf201c17f, 0x54ffff80, 0x72001d5f,
    0x54000060, 0x38401d09, 0x35ffffe9, 0xcb000100, 0x14000000, 0xaa1f03e0, 0x14000000,
];
const START = 100;  // dynarmic_test_reader's start address

function strlenProgram(text, align) {
    // adr x0, <string>; strlen; string data. The reader runs one tick per word, so the trailing
    // zeros also budget enough ticks for the byte loops.
    const dataOffset = 4 * (1 + STRLEN.length);
    const lead = ((8 - ((START + dataOffset) % 8)) % 8) + align;
    const bytes = Buffer.alloc(Math.ceil((lead + text.length + 600) / 4) * 4);
    bytes.write(text, lead, 'latin1');
    const imm = dataOffset + lead;
    const adr = (0x10000000 | ((imm & 3) << 29) | ((imm >> 2) << 5)) >>> 0;
    const words = [adr, ...STRLEN];
    for (let i = 0; i < bytes.length; i += 4) words.push(bytes.readUInt32LE(i));
    return [
        `instructions: ${words.map(w => w.toString(16).padStart(8, '0')).join(' ')}`,
        `initial_regs: ${Array(31).fill('0').join(' ')}`,
        `initial_vecs: ${Array(32).fill('0:0').join(' ')}`,
        'initial_sp: 10000', 'initial_pstate: 0', 'initial_fpcr: 0', '',
    ].join('\n');
}

const finalX0 = out => parseInt(out.match(/^final_regs: (\S+)/m)?.[1] ?? 'NaN', 16);

let failures = 0;

const strlenCases = [];
for (const text of ['', 'a', 'pctl', 'pctl:s', 'fsp-srv', 'abcdefgh', 'abcdefghijklmnop']) {
    for (let align = 0; align < 8; ++align) strlenCases.push({ text, align });
}
const strlenResults = await pool(strlenCases.map(({ text, align }) => async () => {
    const input = strlenProgram(text, align);
    const [interp, jit] = await Promise.all([run('dynarmic_test_reader.js', ['a64'], INTERP, input),
                                             run('dynarmic_test_reader.js', ['a64'], JIT, input)]);
    return { text, align, interp: finalX0(interp), jit: finalX0(jit) };
}));
for (const r of strlenResults) {
    if (r.interp !== r.text.length || r.jit !== r.text.length) {
        failures++;
        console.log(`FAIL strlen("${r.text}") at alignment ${r.align}: interpreter ${r.interp}, jit ${r.jit}`);
    }
}
console.log(`strlen: ${strlenResults.length - failures}/${strlenResults.length} cases match`);

// Random single blocks from dynarmic_test_generator, compared case by case.
const seeds = Array.from({ length: Number(args.seeds) }, (_, i) => i + 1);
const split = out => out.split('\n===\n');
const genResults = await pool(seeds.map(seed => async () => {
    const argv = ['a64', String(seed), '16', args.iterations, 'disas'];
    const [interp, jit] = await Promise.all([run('dynarmic_test_generator.js', argv, INTERP),
                                             run('dynarmic_test_generator.js', argv, JIT)]);
    const a = split(interp);
    const b = split(jit);
    const bad = a.map((c, i) => (c === b[i] ? null : i)).filter(i => i !== null);
    return { seed, total: a.length, bad, first: bad.length ? [a[bad[0]], b[bad[0]]] : null };
}));
for (const r of genResults) {
    if (r.bad.length) {
        failures++;
        console.log(`FAIL generator seed ${r.seed}: ${r.bad.length}/${r.total} cases differ; first:\n` +
            `--- interpreter\n${r.first[0]}\n--- jit\n${r.first[1]}`);
    }
}
console.log(`generator: ${genResults.filter(r => !r.bad.length).length}/${genResults.length} seeds match`);

process.exit(failures ? 1 : 0);

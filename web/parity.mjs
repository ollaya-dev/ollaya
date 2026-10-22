// Web runtime against the native one, on the same requests.
//
//   node web/parity.mjs <model-dir> <requests.jsonl> <out.jsonl>
//
// <model-dir> holds model.onnx, tokenizer.json, decision.json, calibration.json and the weights
// under their blob names (sha256-<hex>), as the native parity example loads them. Each request line
// is {id, state, questions}. The web runtime writes goldens-shaped records (token ids, markers and
// option logits per question) to <out.jsonl>; then the native runner's parity example
// (crates/ollaya-runner/examples/parity.rs) checks them: identical token ids and markers, the same
// decisions, and the calibrated probabilities' max abs difference. Exits non-zero when it fails.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { load, ort } from './ollaya.js';

const [dir, requests, out] = process.argv.slice(2);
if (!out) {
  console.error('usage: node web/parity.mjs <model-dir> <requests.jsonl> <out.jsonl>');
  process.exit(2);
}
const read = (f) => fs.readFileSync(path.join(dir, f));
// model.packed (train/ternary.py) stands in for the weights blob it unpacks to.
const packed = fs.existsSync(path.join(dir, 'model.packed'));
const weights = packed ? ['model.packed'] : fs.readdirSync(dir).filter((f) => f.startsWith('sha256-'));

let t = performance.now();
const model = await load(
  {
    graph: read('model.onnx'),
    tokenizer: read('tokenizer.json'),
    decision: read('decision.json'),
    calibration: read('calibration.json'),
    weights: weights.map((f) => (packed ? { packed: true, data: read(f) } : { path: f, data: read(f) })),
  },
  { wasm: fs.readFileSync(new URL('./pkg/ollaya_web_bg.wasm', import.meta.url)) },
);
const loadMs = performance.now() - t;

const lines = fs.readFileSync(requests, 'utf8').split('\n').filter(Boolean);
const sink = fs.createWriteStream(out);
const times = [];
let questions = 0;
const failed = [];
const rejected = [];
for (const line of lines) {
  const { id, state, questions: qs } = JSON.parse(line);
  t = performance.now();
  let r;
  try {
    r = await model.run({ state, questions: qs });
  } catch (e) {
    // A 4xx is the daemon's own request validation (ollaya_api::validate), which rejects it too.
    (e.status >= 400 && e.status < 500 ? rejected : failed).push(`${id}: ${e.status ?? ''} ${e.message ?? e}`);
    continue;
  }
  times.push(performance.now() - t);
  questions += r.encodings.length;
  const items = r.encodings.map((e, i) => ({
    ids: e.ids,
    markers: e.markers,
    logits: Array.from(r.logits.subarray(i * r.markers, i * r.markers + e.markers.length)),
  }));
  sink.write(JSON.stringify({ id, state, questions: qs, items }) + '\n');
}
await new Promise((resolve) => sink.end(resolve));

times.sort((a, b) => a - b);
const q = (p) => times[Math.min(times.length - 1, Math.floor((times.length * p) / 100))].toFixed(0);
console.log(
  `web: ${times.length} requests, ${questions} questions | load ${loadMs.toFixed(0)} ms | ` +
    `per request p50 ${q(50)} ms, p90 ${q(90)} ms, max ${q(100)} ms | ` +
    `onnxruntime-web ${ort.env.versions.web}, wasm threads ${ort.env.wasm.numThreads ?? 'default'} on ${os.cpus().length} cores`,
);
if (rejected.length) console.log(`web: ${rejected.length} requests rejected by the API contract, as /api/decide rejects them`);
for (const f of failed) console.log(`web failed ${f}`);

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const native = spawnSync(
  'cargo',
  ['run', '--release', '--locked', '-q', '-p', 'ollaya-runner', '--example', 'parity', '--', path.resolve(dir), path.resolve(out)],
  { cwd: root, stdio: 'inherit' },
);
process.exit(failed.length > 0 ? 1 : (native.status ?? 1));

// Ollaya encoder decision models (layout laya-markers-v1) in the browser.
//
//   import { fromManifest, load } from './ollaya.js';
//   const model = await load(fromManifest(manifest));
//   const response = await model.decide({ state, questions });   // the /api/decide body
//
// Tokenizer, layout, calibration and answers run in WebAssembly built from crates/ollaya-web (the
// native runner's own code); the graph runs in onnxruntime-web. Weights are never re-hosted: the
// manifest's weights layer points at the author's Hugging Face file, fetched as it is.

import * as ort from 'onnxruntime-web';
import init, { Model } from './pkg/ollaya_web.js';

export { ort };

const MEDIA = {
  graph: 'application/vnd.ollaya.graph.onnx',
  weights: 'application/vnd.ollaya.weights',
  tokenizer: 'application/vnd.ollaya.tokenizer',
  decision: 'application/vnd.ollaya.decision',
  calibration: 'application/vnd.ollaya.calibration',
};

/** The files `load` takes, as URLs, from a registry manifest (registry/v2/library/<name>/manifests/<tag>). */
export function fromManifest(manifest, precision = 'fp32') {
  const layers = manifest.layers;
  const one = (type) => layers.find((l) => l.mediaType === type);
  const graph = layers.find(
    (l) => l.mediaType === MEDIA.graph && (l.annotations?.['org.ollaya.precision'] ?? 'fp32') === precision,
  );
  if (!graph) throw new Error(`the manifest has no ${precision} graph`);
  return {
    graph: graph.urls[0],
    // The graph names each weights file by its blob name, `sha256-<hex>`, as the native store does.
    weights: layers
      .filter((l) => l.mediaType === MEDIA.weights)
      .map((l) => ({ path: l.digest.replace(':', '-'), data: l.urls[0] })),
    tokenizer: one(MEDIA.tokenizer).urls[0],
    decision: one(MEDIA.decision).urls[0],
    calibration: one(MEDIA.calibration)?.urls[0],
  };
}

// A file is a URL (string) or its contents (Uint8Array / ArrayBuffer).
async function bytes(file) {
  if (typeof file !== 'string') return new Uint8Array(file);
  const res = await fetch(file);
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function text(file) {
  return file === undefined ? '' : new TextDecoder().decode(await bytes(file));
}

/**
 * Load a model from its files: `{graph, weights: [{path, data}], tokenizer, decision, calibration}`.
 * Options: `name` (the model name answers carry, e.g. 'largha:150m'), `wasm` (ollaya_web_bg.wasm as a
 * URL or bytes; default next to pkg/ollaya_web.js), `device` ('wasm' or 'webgpu'), `sessionOptions`
 * (merged into onnxruntime-web's).
 */
export async function load(files, { name = 'web', wasm, device = 'wasm', sessionOptions = {} } = {}) {
  await init(wasm === undefined ? undefined : { module_or_path: wasm });
  // Weights are fetched here, not by onnxruntime-web, which reads a string as a file path under Node.
  const [tokenizer, decision, calibration, graph, externalData] = await Promise.all([
    text(files.tokenizer),
    text(files.decision),
    text(files.calibration),
    bytes(files.graph),
    Promise.all((files.weights ?? []).map(async (w) => ({ path: w.path, data: await bytes(w.data) }))),
  ]);
  const model = new Model(tokenizer, decision, calibration);
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    // As the native runner (GraphOptimizationLevel::Level3).
    graphOptimizationLevel: 'all',
    externalData,
    ...sessionOptions,
  });
  return new Decider(name, model, session);
}

// The wasm side throws the daemon's error body ({status, error, code, detail?}) as JSON.
function apiError(e) {
  let body;
  try {
    body = JSON.parse(e.message);
  } catch {
    return e;
  }
  return Object.assign(new Error(body.error), { status: body.status, body });
}

export class Decider {
  constructor(name, model, session) {
    this.name = name;
    this.model = model;
    this.session = session;
  }

  /**
   * `/api/decide`: `{state, questions, extras?}` to the daemon's response body. Invalid requests throw
   * an Error with the daemon's `status` and error `body`.
   */
  async decide(request) {
    return (await this.run(request)).response;
  }

  /** `decide`, with what parity checks read: each row's encoding and the raw option logits [rows, markers]. */
  async run(request) {
    const started = performance.now();
    let batch;
    try {
      batch = this.model.encode(JSON.stringify({ model: this.name, ...request }));
    } catch (e) {
      throw apiError(e);
    }
    try {
      const { rows, seq, markers } = batch;
      const inputs = {
        input_ids: () => new ort.Tensor('int64', batch.input_ids(), [rows, seq]),
        attention_mask: () => new ort.Tensor('int64', batch.attention_mask(), [rows, seq]),
        marker_pos: () => new ort.Tensor('int64', batch.marker_pos(), [rows, markers]),
        marker_mask: () => new ort.Tensor('bool', batch.marker_mask(), [rows, markers]),
        qtype: () => new ort.Tensor('int64', batch.qtype(), [rows]),
      };
      const feeds = {};
      for (const name of this.session.inputNames) {
        if (!inputs[name]) throw new Error(`the graph takes an input this runtime does not build: ${name}`);
        feeds[name] = inputs[name]();
      }
      const evalStarted = performance.now();
      const out = await this.session.run(feeds);
      const evalNs = Math.round((performance.now() - evalStarted) * 1e6);
      const logits = out.logits.data;
      const act = out.act_logits?.data ?? new Float32Array();
      const response = JSON.parse(this.model.answer(batch, logits, act));
      Object.assign(response, {
        created_at: new Date().toISOString(),
        total_duration: Math.round((performance.now() - started) * 1e6),
        load_duration: 0,
        eval_duration: evalNs,
      });
      return { encodings: JSON.parse(batch.encodings()), markers, logits, response };
    } finally {
      batch.free();
    }
  }
}

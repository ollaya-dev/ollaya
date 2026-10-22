# @ollaya/web

Ollaya's decision models in a web page. Ask questions about a state (a message, a ticket, a JSON object) and get
calibrated answers, computed on the visitor's device: nothing is sent to a server.

It takes the same request and gives the same answer as Ollaya's `/api/decide`, and it gives the same numbers: the
tokenizer, the question layout, calibration and answers are Ollaya's own Rust code compiled to WebAssembly, the
graph runs in [onnxruntime-web](https://onnxruntime.ai), and every release is checked against the native runtime
(`parity.mjs`).

Try it: [ollaya.dev/try](https://ollaya.dev/try).

## Install

```sh
npm install @ollaya/web
```

```js
import { fromHub, load } from '@ollaya/web';

const model = await load(fromHub('<org>/<model>'), {
  onProgress: ({ loaded, total }) => console.log(`${loaded} of ${total} bytes`),
});
const { answers } = await model.decide({
  state: "I was charged twice for order A-104. Refund the duplicate today or I'm cancelling.",
  questions: {
    refund: { type: 'noul', instructions: 'Is the customer asking for a refund?' },
    team: { type: 'choice', instructions: 'Which team handles this?',
            criteria: { billing: 'Payments, charges, refunds', technical: 'Bugs, outages, errors' } },
    urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['Can wait', 'This week', 'Today'] },
  },
});
// answers.refund   → { type: 'noul', noul: 0.98 }
// answers.team     → { type: 'choice', choice: 'billing', confidence: …, probabilities: { billing: 0.98, … } }
// answers.urgency  → { type: 'score', score: 1.6, legend: { … }, probabilities: { … } }
```

Without a bundler, one page:

```html
<script type="importmap">
  { "imports": { "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.bundle.min.mjs" } }
</script>
<script type="module">
  import { fromHub, load } from 'https://cdn.jsdelivr.net/npm/@ollaya/web@0.1.0/ollaya.js';
  const model = await load(fromHub('<org>/<model>'));
  console.log(await model.decide({ state: 'Where is my parcel?', questions: { q: { type: 'noul', instructions: 'Is this about shipping?' } } }));
</script>
```

## API

- `fromHub(repo, { revision = 'main', endpoint })`: the files of one of Ollaya's ternary models in its Hugging
  Face repository (`onnx/model.onnx`, with the weights at 2 and 4 bits inside it, plus `tokenizer.json`,
  `decision.json` and `calibration.json`).
- `fromManifest(manifest, precision = 'fp32')`: the files of any Ollaya registry model whose layout is
  `laya-markers-v1`, from its manifest.
- `load(files, { name, wasm, onProgress, sessionOptions })`: downloads and starts the model. `files` holds URLs or
  bytes. `wasm` is where `ollaya_web_bg.wasm` is, if your bundler moves it.
- `model.decide(request)`: the `/api/decide` body (`state`, `questions`, optional `extras`) to its response. An
  invalid request throws an `Error` with the daemon's `status` and error `body`.

Questions are the API's three types: `noul` (the probability of yes), `choice` (one of up to 255 described options)
and `score` (2 to 10 described levels). See [the API reference](https://ollaya.dev/docs/api).

## Speed and memory

The ternary 150M model is a 63 MB download plus a 3.6 MB tokenizer. On a laptop CPU it answers a request of a few
questions in 0.2–0.4 s with four WebAssembly threads, using about 0.5 GB of memory. Threads need the page to be
cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
credentialless` or `require-corp`); without them onnxruntime-web uses one thread. In Chrome the first few calls run
slower while the browser optimizes the WebAssembly; a couple of warm-up calls after `load` take care of it.

Files at a commit (`fromHub(repo, { revision: '<commit sha>' })`, and every registry manifest's URLs) never
change, so the browser keeps them in Cache Storage (`ollaya-web`) and the next visit loads without downloading.
`main` is downloaded every time.

## Build from source

`pkg/` is built from `crates/ollaya-web` (`sh build.sh`; needs the `wasm32-unknown-unknown` Rust target and
`wasm-bindgen-cli` 0.2.128). `node parity.mjs <model-dir> <requests.jsonl> <out.jsonl>` checks this runtime
against the native one.

Apache-2.0.

// A fake backend for `npm run preview`: the real library, invented local state and answers.
import type { Backend, LibraryModel, LocalModel, PullProgress, Status } from './backend'

const library: LibraryModel[] = [
  {
    name: 'laya',
    description:
      'Open decision models from Convai Innovations. Typed, calibrated answers to choice, score and yes/no questions in a single forward pass, in English and 100+ languages.',
    caps: ['multilingual', 'router', 'guardrails', 'fine-tuned'],
    tags: [
      { name: 'laya:latest', summary: 'Router: English to laya:en, other languages to laya:multilingual.' },
      { name: 'laya:en', summary: 'English. Best for guardrails and email triage.' },
      { name: 'laya:multilingual', summary: '100+ languages, 1024-token context.' },
      { name: 'laya:typed-decisions', summary: 'Fine-tuned on typed-decisions workflows.' },
      { name: 'laya:en-fp16', summary: '' },
    ],
  },
  {
    name: 'decider',
    description: 'Decoder decision models by Mapika on Qwen3.5. The most accurate open decision model Ollaya ships.',
    caps: ['decoder'],
    tags: [
      { name: 'decider:latest', summary: 'decider:2b.' },
      { name: 'decider:0.8b', summary: 'Smaller and faster.' },
    ],
  },
  { name: 'nli', description: 'Zero-shot classifiers by Moritz Laurer.', caps: ['zero-shot'], tags: [{ name: 'nli:latest', summary: '' }] },
  { name: 'gliclass', description: 'Instruction-following zero-shot classifier by Knowledgator.', caps: ['zero-shot'], tags: [{ name: 'gliclass:latest', summary: '' }] },
]

export function mock(): Backend {
  let status: Status = { running: true, version: '0.4.0', url: 'http://127.0.0.1:11435' }
  let local: LocalModel[] = [
    { name: 'laya:latest', size: 11_000 },
    { name: 'laya:en', size: 854_000_000 },
    { name: 'laya:multilingual', size: 684_000_000 },
  ]
  let listener: ((p: PullProgress) => void) | null = null
  const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))
  return {
    status: async () => status,
    startServer: async () => ((status = { ...status, running: true, version: '0.4.0' }), status),
    stopServer: async () => ((status = { ...status, running: false, version: null }), status),
    library: async () => library,
    installed: async () => (status.running ? local : []),
    pull: async (model) => {
      const total = 1_500_000_000
      for (let done = 0; done <= total; done += total / 20) {
        listener?.({ model, status: 'pulling', completed: done, total, done: false, error: null })
        await wait(120)
      }
      local = [...local, { name: model.includes(':') ? model : `${model}:latest`, size: total }]
      listener?.({ model, status: 'success', completed: total, total, done: true, error: null })
    },
    remove: async (model) => {
      local = local.filter((m) => m.name !== model)
    },
    presets: async () =>
      ['triage', 'email', 'guard', 'moderation', 'router'].map((name) => ({ name, questions: { intent: { type: 'choice' } } })),
    decide: async (model) => {
      await wait(200)
      return {
        model: model === 'laya' || model === 'laya:latest' ? 'laya:en' : model,
        routing: { model: 'laya:en' },
        total_duration: 8_900_000,
        answers: {
          intent: { type: 'choice', choice: 'refund', confidence: 0.999 },
          is_urgent: { type: 'noul', noul: 0.13 },
          frustration: { type: 'score', score: 1.59, confidence: 0.31, legend: { '0': 'calm and neutral', '1': 'concerned but civil', '2': 'clearly annoyed', '3': 'very angry' } },
          refund_requested: { type: 'noul', noul: 0.88 },
          churn_risk: { type: 'noul', noul: 0.89 },
        },
      }
    },
    onPullProgress: (handler) => {
      listener = handler
    },
  }
}

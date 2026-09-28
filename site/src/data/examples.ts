import type { CodeTab } from '../components/ui'
import { LOCAL_API } from '../site'

/** The support-triage example used across the site (hero, usage boxes). */
export const TRIAGE_STATE = 'I was charged twice for my subscription this month and want a refund.'

export const triageQuestions = {
  department: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: {
      billing: 'Payments, invoices and refunds',
      technical: 'Bugs, errors and outages',
      account: 'Login, profile and settings',
    },
  },
  refund: {
    type: 'noul',
    instructions: 'Is the customer asking for a refund?',
  },
}

const indent = (text: string, pad: string) =>
  text
    .split('\n')
    .map((line, i) => (i === 0 ? line : pad + line))
    .join('\n')

/**
 * CLI / cURL / Python / JavaScript snippets for a model reference such as "laya" or "laya:en".
 * A model with built-in questions (`builtin`) is asked about the state alone.
 */
export function usageTabs(ref: string, state: string = TRIAGE_STATE, builtin = false): CodeTab[] {
  const body = builtin ? { model: ref, state } : { model: ref, state, questions: triageQuestions }
  const json2 = JSON.stringify(body, null, 2)
  const json4 = JSON.stringify(body, null, 4)
  const jsBody = JSON.stringify(body, null, 2).replace(/^(\s*)"([a-z_]+)":/gm, '$1$2:')
  const pyRead = builtin ? 'print(answers)' : 'print(answers["department"]["choice"], answers["refund"]["noul"])'
  const jsRead = builtin ? 'console.log(answers);' : 'console.log(answers.department.choice, answers.refund.noul);'

  return [
    {
      key: 'cli',
      label: 'CLI',
      code: builtin ? `ollaya run ${ref} "${state}"` : `ollaya run ${ref} --preset triage "${state}"`,
    },
    {
      key: 'curl',
      label: 'cURL',
      code: `curl ${LOCAL_API}/api/decide \\
  -H "Content-Type: application/json" \\
  -d '${indent(json2, '  ')}'`,
    },
    {
      key: 'python',
      label: 'Python',
      code: `# Already using a TypeSafe SDK? Set TYPESAFE_BASE_URL=${LOCAL_API} instead.
import requests

response = requests.post(
    "${LOCAL_API}/api/decide",
    json=${indent(json4, '    ')},
)
answers = response.json()["answers"]
${pyRead}`,
    },
    {
      key: 'javascript',
      label: 'JavaScript',
      code: `const response = await fetch("${LOCAL_API}/api/decide", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(${indent(jsBody, '  ')}),
});
const { answers } = await response.json();
${jsRead}`,
    },
  ]
}

const VISION_STATE = 'A photo from the warehouse camera.'
const visionQuestions = {
  blocked: { type: 'noul', instructions: 'Is the aisle blocked?' },
  fill: { type: 'score', instructions: 'How full is the shelf?', criteria: ['empty', 'half full', 'full'] },
}

/** Snippets for a vision model: the same request plus one PNG in `images` (base64). */
export function visionUsageTabs(ref: string): CodeTab[] {
  const questions = JSON.stringify(visionQuestions)
  const q2 = indent(JSON.stringify(visionQuestions, null, 2), '  ')
  const q4 = indent(JSON.stringify(visionQuestions, null, 4), '        ')
  return [
    {
      key: 'cli',
      label: 'CLI',
      code: `ollaya run ${ref} --image shelf.png --questions '${questions}' "${VISION_STATE}"`,
    },
    {
      key: 'curl',
      label: 'cURL',
      code: `curl ${LOCAL_API}/api/decide \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${ref}",
  "state": "${VISION_STATE}",
  "images": ["'"$(base64 -w0 shelf.png)"'"],
  "questions": ${indent(q2, '  ')}
}'`,
    },
    {
      key: 'python',
      label: 'Python',
      code: `import base64
import requests

image = base64.b64encode(open("shelf.png", "rb").read()).decode()
response = requests.post(
    "${LOCAL_API}/api/decide",
    json={
        "model": "${ref}",
        "state": "${VISION_STATE}",
        "images": [image],
        "questions": ${q4},
    },
)
answers = response.json()["answers"]
print(answers["blocked"]["noul"], answers["fill"]["score"])`,
    },
    {
      key: 'javascript',
      label: 'JavaScript',
      code: `import { readFile } from "node:fs/promises";

const image = (await readFile("shelf.png")).toString("base64");
const response = await fetch("${LOCAL_API}/api/decide", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "${ref}",
    state: "${VISION_STATE}",
    images: [image],
    questions: ${indent(JSON.stringify(visionQuestions, null, 2), '    ')},
  }),
});
const { answers } = await response.json();
console.log(answers.blocked.noul, answers.fill.score);`,
    },
  ]
}

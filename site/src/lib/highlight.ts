/**
 * Build-time syntax highlighting for the site's code blocks: shell, JSON, Python, JavaScript and
 * Modelfile/Dockerfile. It wraps tokens in `<span class="tok-…">`; the colours are theme tokens in
 * app.css, so they follow light and dark mode. Nothing runs in the browser and there is no
 * dependency: the snippets are small and hand-written, so a few regular expressions per language
 * are enough.
 *
 * Imported by the pages (through esbuild) and by scripts/gen-content.mjs (through Node's type
 * stripping), so this file must stay self-contained and use only erasable TypeScript.
 */

type Kind =
  | 'comment'
  | 'string'
  | 'number'
  | 'literal'
  | 'keyword'
  | 'key'
  | 'fn'
  | 'cmd'
  | 'flag'
  | 'var'
  | 'punct'

const escape = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const tok = (kind: Kind, text: string) => `<span class="tok-${kind}">${escape(text)}</span>`

/** Returns the HTML of `code` highlighted as `lang` (the Markdown fence name); unknown languages are only escaped. */
export function highlight(code: string, lang: string | undefined): string {
  switch ((lang ?? '').toLowerCase()) {
    case 'json':
      return json(code)
    case 'shell':
    case 'sh':
    case 'bash':
    case 'console':
    case 'cli':
    case 'curl':
    case 'powershell':
    case 'ps1':
      return shell(code)
    case 'python':
    case 'py':
      return python(code)
    case 'javascript':
    case 'js':
    case 'typescript':
    case 'ts':
      return javascript(code)
    case 'dockerfile':
    case 'modelfile':
      return modelfile(code)
    case 'http':
      return http(code)
    default:
      return escape(code)
  }
}

// ---------------------------------------------------------------------------------------------

const JSON_RULES: [RegExp, (m: string) => string][] = [
  [/"(?:[^"\\\n]|\\.)*"(?=\s*:)/y, (m) => tok('key', m)],
  [/"(?:[^"\\\n]|\\.)*"/y, (m) => tok('string', m)],
  [/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y, (m) => tok('number', m)],
  [/\b(?:true|false|null)\b/y, (m) => tok('literal', m)],
  [/[{}[\],:]/y, (m) => tok('punct', m)],
]

function json(code: string): string {
  return scan(code, JSON_RULES)
}

/** Runs `rules` (sticky regexes, first match wins) over `code`; anything unmatched is copied through. */
function scan(code: string, rules: [RegExp, (m: string) => string][]): string {
  let out = ''
  let i = 0
  outer: while (i < code.length) {
    for (const [re, emit] of rules) {
      re.lastIndex = i
      const m = re.exec(code)
      if (m && m[0].length > 0) {
        out += emit(m[0])
        i += m[0].length
        continue outer
      }
    }
    out += escape(code[i]!)
    i += 1
  }
  return out
}

// ---------------------------------------------------------------------------------------------

const SHELL_KEYWORDS = new Set(['export', 'sudo', 'if', 'then', 'else', 'fi', 'for', 'do', 'done', 'in', 'echo', 'cd'])

/** Shell: commands, flags, variables, strings and comments. A single-quoted JSON body (curl -d '{…}') is highlighted as JSON. */
function shell(code: string): string {
  let out = ''
  let i = 0
  // At the start of a command, where the next word is the program (or an assignment).
  let start = true
  while (i < code.length) {
    const rest = code.slice(i)
    const prev = i === 0 ? '\n' : code[i - 1]!
    let m: RegExpMatchArray | null
    if (rest[0] === '#' && /\s/.test(prev)) {
      const line = rest.match(/^#[^\n]*/)![0]
      out += tok('comment', line)
      i += line.length
    } else if ((m = rest.match(/^\\\n/))) {
      out += tok('punct', '\\') + '\n'
      i += 2
    } else if (rest[0] === '\n') {
      out += '\n'
      start = true
      i += 1
    } else if (/[ \t]/.test(rest[0]!)) {
      const ws = rest.match(/^[ \t]+/)![0]
      out += ws
      i += ws.length
    } else if ((m = rest.match(/^'([^']*)'/))) {
      const inner = m[1]!
      out += /^\s*[{[]/.test(inner)
        ? tok('string', "'") + json(inner) + tok('string', "'")
        : tok('string', m[0])
      i += m[0].length
      start = false
    } else if ((m = rest.match(/^"(?:[^"\\]|\\.)*"/))) {
      out += tok('string', m[0])
      i += m[0].length
      start = false
    } else if ((m = rest.match(/^(?:\$\{[^}\n]*\}|\$[A-Za-z_][A-Za-z0-9_]*)/))) {
      out += tok('var', m[0])
      i += m[0].length
      start = false
    } else if ((m = rest.match(/^(?:&&|\|\||[|;&])/))) {
      out += tok('punct', m[0])
      i += m[0].length
      start = true
    } else if (start && (m = rest.match(/^([A-Za-z_][A-Za-z0-9_]*)=/))) {
      // NAME=value before (or instead of) a command.
      out += tok('var', m[1]!) + tok('punct', '=')
      i += m[0].length
      const value = code.slice(i).match(/^[^\s'"]*/)![0]
      out += escape(value)
      i += value.length
    } else if ((m = rest.match(/^--?[A-Za-z0-9][\w-]*(?:=)?/)) && /\s/.test(prev)) {
      out += tok('flag', m[0])
      i += m[0].length
      start = false
    } else {
      const word = rest.match(/^[^\s'"|;&$]+/)?.[0] ?? rest[0]!
      if (start) {
        out += SHELL_KEYWORDS.has(word) ? tok('keyword', word) : tok('cmd', word)
        // `export NAME=…` and `sudo cmd`: the next word is still an assignment or a command.
        start = word === 'export' || word === 'sudo'
      } else {
        out += escape(word)
      }
      i += word.length
    }
  }
  return out
}

// ---------------------------------------------------------------------------------------------

const PY_KEYWORDS = new Set([
  'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
  'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not',
  'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
])
const PY_LITERALS = new Set(['None', 'True', 'False'])

function python(code: string): string {
  return code_like(code, {
    comment: /#[^\n]*/y,
    strings: [/[rbfuRBFU]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?''')/y, /[rbfuRBFU]{0,2}(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/y],
    keywords: PY_KEYWORDS,
    literals: PY_LITERALS,
    // `json=` inside a call is a keyword argument.
    keyArg: true,
  })
}

const JS_KEYWORDS = new Set([
  'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'default', 'else', 'export',
  'extends', 'finally', 'for', 'from', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new', 'of',
  'return', 'switch', 'throw', 'try', 'typeof', 'var', 'while',
])
const JS_LITERALS = new Set(['true', 'false', 'null', 'undefined'])

function javascript(code: string): string {
  return code_like(code, {
    comment: /\/\/[^\n]*|\/\*[\s\S]*?\*\//y,
    strings: [/`(?:[^`\\]|\\.)*`/y, /"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/y],
    keywords: JS_KEYWORDS,
    literals: JS_LITERALS,
    // `model:` in an object literal is a property name.
    keyColon: true,
  })
}

interface CodeLike {
  comment: RegExp
  strings: RegExp[]
  keywords: Set<string>
  literals: Set<string>
  keyArg?: boolean
  keyColon?: boolean
}

/** Python and JavaScript: identifiers are classified by what follows them. */
function code_like(code: string, lang: CodeLike): string {
  const rules: [RegExp, (m: string) => string][] = [
    [lang.comment, (m) => tok('comment', m)],
    ...lang.strings.map((re): [RegExp, (m: string) => string] => [re, (m) => tok('string', m)]),
    [/\b\d+(?:\.\d+)?\b/y, (m) => tok('number', m)],
    [/[A-Za-z_$][\w$]*/y, (m) => m],
    [/[{}[\]().,:;=+\-*/<>!?|&]/y, (m) => tok('punct', m)],
  ]
  let out = ''
  let i = 0
  let prevSig = ''
  outer: while (i < code.length) {
    for (const [re, emit] of rules) {
      re.lastIndex = i
      const m = re.exec(code)
      if (!m || m[0].length === 0) continue
      const text = m[0]
      i += text.length
      if (/^[A-Za-z_$]/.test(text) && emit(text) === text) {
        const after = code.slice(i)
        if (lang.keywords.has(text)) out += tok('keyword', text)
        else if (lang.literals.has(text)) out += tok('literal', text)
        else if (/^\s*\(/.test(after)) out += tok('fn', text)
        else if (lang.keyArg && /^=(?!=)/.test(after) && (prevSig === '(' || prevSig === ',')) out += tok('key', text)
        else if (lang.keyColon && /^\s*:(?!:)/.test(after) && (prevSig === '{' || prevSig === ',')) out += tok('key', text)
        else out += escape(text)
      } else {
        out += emit(text)
      }
      if (text.trim()) prevSig = text.trim().slice(-1)
      continue outer
    }
    const ch = code[i]!
    out += escape(ch)
    if (ch.trim()) prevSig = ch
    i += 1
  }
  return out
}

// ---------------------------------------------------------------------------------------------

/** An endpoint line such as `POST /api/decide`. */
function http(code: string): string {
  return code
    .split('\n')
    .map((line) => {
      const m = line.match(/^([A-Z]+)( +)(\S+)(.*)$/)
      return m ? tok('keyword', m[1]!) + m[2]! + tok('string', m[3]!) + escape(m[4]!) : escape(line)
    })
    .join('\n')
}

// ---------------------------------------------------------------------------------------------

/** Modelfile (and Dockerfile): the instruction starting each line, comments, and quoted strings. */
function modelfile(code: string): string {
  return code
    .split('\n')
    .map((line) => {
      if (/^\s*#/.test(line)) return tok('comment', line)
      const m = line.match(/^(\s*)([A-Z]+)(\b.*)$/)
      if (!m) return escape(line)
      const rest = m[3]!.replace(/"[^"]*"|[^"]+/g, (part) => (part.startsWith('"') ? tok('string', part) : escape(part)))
      return m[1]! + tok('keyword', m[2]!) + rest
    })
    .join('\n')
}

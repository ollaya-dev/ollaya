#!/usr/bin/env bash
# End-to-end smoke test of a released Ollaya on this machine, the way a new user meets it:
# install, pull, run, the HTTP API, a Modelfile, error paths, and request latency.
#
#   bash scripts/smoke.sh [model ...]        (default: laya nli gliclass decider:0.8b)
#
# Settings (environment):
#   OLLAYA_SITE=https://ollaya.dev   where install.sh comes from
#   SMOKE_SKIP_INSTALL=1                     test the ollaya already on PATH
#   SMOKE_N=30                               requests per latency measurement
#
# `laya` is always tested in full; the other models get pull, a cold run and latency.
# Leaves the daemon running and the models pulled. Exits non-zero if any check failed.
set -u

SITE=${OLLAYA_SITE:-https://ollaya.dev}
N=${SMOKE_N:-30}
API=http://127.0.0.1:11435
if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(laya nli gliclass decider:0.8b); fi
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ollaya-smoke.XXXXXX")
PASS=0
FAIL=0

EN="My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
TR="Siparişim hâlâ gelmedi ve destek ekibi cevap vermiyor. Bugün paramı iade edin yoksa rakibinize geçiyorum."

now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }
elapsed() { perl -e "printf '%.2f', $2 - $1"; }
report() { # report <ok|FAIL> <name> <detail>
    if [ "$1" = ok ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
    printf '%-5s %-46s %s\n' "$1" "$2" "$3"
}
# step <name> <command...>: run, time and record a command; its output is kept in $WORK/out.
step() {
    local name=$1 t0 rc
    shift
    t0=$(now)
    "$@" >"$WORK/out" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        report ok "$name" "$(elapsed "$t0" "$(now)") s"
    else
        report FAIL "$name" "$(elapsed "$t0" "$(now)") s, exit $rc"
        tail -n 15 "$WORK/out" | sed 's/^/      | /'
    fi
    return $rc
}
# expect <name> <pattern>: the last step's output must match the extended regex.
expect() {
    if grep -Eq "$2" "$WORK/out"; then
        report ok "$1" ""
    else
        report FAIL "$1" "no match for /$2/"
        tail -n 15 "$WORK/out" | sed 's/^/      | /'
    fi
}
# note <name> <pattern>: like expect, but a miss is a quality note (zero-shot models may disagree), not a failure.
note() {
    if grep -Eq "$2" "$WORK/out"; then report ok "$1" ""; else printf '%-5s %-46s %s\n' note "$1" "$(grep -E '^intent' "$WORK/out" | head -n 1)"; fi
}
# latency <name> <body-file>: N POSTs to /v1/systemone; all must return 200.
latency() {
    local codes
    for _ in $(seq 1 "$N"); do
        curl -s -o /dev/null -w '%{http_code} %{time_total}\n' -H 'Content-Type: application/json' \
            --data-binary "@$2" "$API/v1/systemone"
    done >"$WORK/lat"
    codes=$(awk '$1 != 200' "$WORK/lat" | wc -l | tr -d ' ')
    local stats
    stats=$(awk '{print $2}' "$WORK/lat" | sort -n | awk '{a[NR] = $1} END {
        i50 = int(NR * 0.50 + 0.5); i95 = int(NR * 0.95 + 0.5); if (i95 > NR) i95 = NR
        printf "p50 %.1f ms, p95 %.1f ms, min %.1f ms (n=%d)", a[i50] * 1000, a[i95] * 1000, a[1] * 1000, NR }')
    if [ "$codes" = 0 ]; then report ok "$1" "$stats"; else report FAIL "$1" "$codes non-200 responses; $stats"; fi
}
body() { # body <file> <model> <state text>
    python3 - "$@" "$WORK/triage.json" <<'EOF'
import json, sys
out, model, text, questions = sys.argv[1:5]
json.dump({"model": model, "state": {"message": text}, "questions": json.load(open(questions))},
          open(out, "w"), ensure_ascii=False)
EOF
}

cat >"$WORK/triage.json" <<'EOF'
{
  "intent": {"type": "choice", "instructions": "What does the customer want in `message`?",
    "criteria": {"refund": "money returned or a duplicate charge reversed",
      "technical_help": "a bug, outage or integration problem",
      "billing_question": "a question about an invoice, plan or payment method",
      "information": "general information, pricing or how-to",
      "cancellation": "wants to cancel or downgrade", "other": "none of the other options fits"}},
  "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
  "frustration": {"type": "score", "instructions": "How frustrated does the customer sound in `message`?",
    "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry or using strong language"]},
  "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
  "churn_risk": {"type": "noul", "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?"}
}
EOF

echo "== $(hostname) · $(uname -sm) · $(date '+%Y-%m-%d %H:%M')"

# --- install -------------------------------------------------------------------------------

if [ -z "${SMOKE_SKIP_INSTALL:-}" ]; then
    step "install ($SITE/install.sh)" sh -c "curl -fsSL '$SITE/install.sh' | sh"
    grep -E '^>>>|WARNING|ERROR' "$WORK/out" | sed 's/^/      /'
fi
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
command -v ollaya >/dev/null || { report FAIL "ollaya on PATH" "not found"; exit 1; }
step "ollaya -v" ollaya -v && sed 's/^/      /' "$WORK/out"

# --- laya: CLI -----------------------------------------------------------------------------

step "pull laya" ollaya pull laya
step "run laya, cold (starts daemon, loads model)" ollaya run laya --preset triage "$EN"
expect "  answers intent=refund" 'intent +refund'
step "run laya, warm" ollaya run laya --preset triage "$EN"
step "run laya, Turkish, --format json" ollaya run laya --preset triage --format json "$TR"
expect "  routed to laya:multilingual" 'laya:multilingual'
step "list" ollaya list
expect "  lists laya" 'laya'
step "ps" ollaya ps
sed 's/^/      /' "$WORK/out"
step "show laya" ollaya show laya

# --- laya: HTTP API ------------------------------------------------------------------------

step "GET /api/version" curl -fsS "$API/api/version"
step "GET /v1/models" curl -fsS "$API/v1/models"
body "$WORK/en.json" laya "$EN"
body "$WORK/tr.json" laya "$TR"
step "POST /v1/systemone" sh -c "curl -fsS -H 'Content-Type: application/json' --data-binary @'$WORK/en.json' '$API/v1/systemone'"
expect "  TypeSafe answer for intent" '"intent"'
step "POST /api/decide" sh -c "curl -fsS -H 'Content-Type: application/json' --data-binary @'$WORK/tr.json' '$API/api/decide'"
expect "  routing reported" '"routing"'
latency "latency laya, English, 5 questions" "$WORK/en.json"
latency "latency laya, Turkish, 5 questions" "$WORK/tr.json"

# --- Modelfile, cp, rm, errors -------------------------------------------------------------

printf 'FROM laya\nQUESTIONS ./triage.json\n' >"$WORK/Modelfile"
step "create smoke-triage -f Modelfile" sh -c "cd '$WORK' && ollaya create smoke-triage -f Modelfile"
step "run smoke-triage (built-in questions)" ollaya run smoke-triage "$EN"
expect "  answers intent=refund" 'intent +refund'
step "cp smoke-triage smoke-copy" ollaya cp smoke-triage smoke-copy
step "rm smoke-triage smoke-copy" ollaya rm smoke-triage smoke-copy
if ollaya run no-such-model --preset triage "x" >"$WORK/out" 2>&1; then
    report FAIL "run unknown model fails" "exit 0"
else
    report ok "run unknown model fails" "$(head -n 1 "$WORK/out")"
fi
printf '{"model": "laya", "state": {"m": "x"}, "questions": {"q": {"type": "choice"}}}' >"$WORK/bad.json"
code=$(curl -s -o "$WORK/out" -w '%{http_code}' -H 'Content-Type: application/json' --data-binary "@$WORK/bad.json" "$API/v1/systemone")
case $code in
    4??) report ok "invalid question → $code" "$(head -c 120 "$WORK/out")" ;;
    *) report FAIL "invalid question → 4xx" "got $code: $(head -c 200 "$WORK/out")" ;;
esac

# --- other models --------------------------------------------------------------------------

for m in "${MODELS[@]}"; do
    [ "$m" = laya ] && continue
    step "pull $m" ollaya pull "$m" || continue
    step "run $m, cold (loads model)" ollaya run "$m" --preset triage "$EN"
    note "  answers intent=refund" 'intent +refund'
    body "$WORK/m.json" "$m" "$EN"
    latency "latency $m, 5 questions" "$WORK/m.json"
done

step "ps" ollaya ps
sed 's/^/      /' "$WORK/out"
for m in laya "${MODELS[@]}"; do ollaya stop "$m" >/dev/null 2>&1; done
echo "== $(hostname): $PASS passed, $FAIL failed"
rm -rf "$WORK"
[ "$FAIL" -eq 0 ]

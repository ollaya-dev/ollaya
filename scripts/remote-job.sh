#!/usr/bin/env bash
# Send a command to a GPU machine (hope-wsl, choso-wsl) and wait for it. The command runs there in a
# detached tmux session, so a dropped ssh connection or this Mac going to sleep does not kill it.
#
#   scripts/remote-job.sh <host> [-n name] [-d remote-dir] [-i seconds] [--detach] -- <command...>
#   scripts/remote-job.sh <host> [-i seconds] --wait <name>
#
# The words after -- are joined and run by the remote shell from <remote-dir> (relative to the
# remote home, default the home), so quote a pipe or `;` as one word. The job is tmux session
# `ollaya-<name>` (<name> defaults to a timestamp); its output goes to ~/ollaya-jobs/<name>.log on
# the remote and its exit code to ~/ollaya-jobs/<name>.exit when it ends. Until that file appears
# the script tails the log and polls every <seconds> (default 30), then exits with the command's
# exit code. --detach returns right after starting and prints how to attach; --wait picks a job
# up again, also after this script was interrupted. Uses the ssh aliases of ~/.ssh/config as they
# are and installs nothing: tmux must already be on the remote. Which machine does what: docs/remote-jobs.md.
set -euo pipefail

usage() {
    echo "usage: $0 <host> [-n name] [-d remote-dir] [-i seconds] [--detach] -- <command...>" >&2
    echo "       $0 <host> [-i seconds] --wait <name>" >&2
    exit 2
}

[ $# -ge 1 ] || usage
host=$1
shift
name=$(date +%Y%m%d-%H%M%S) dir=. interval=30 detach='' wait=''
while [ $# -gt 0 ]; do
    case $1 in
        -n) name=${2:?-n needs a name}; shift 2 ;;
        -d) dir=${2:?-d needs a directory}; shift 2 ;;
        -i) interval=${2:?-i needs seconds}; shift 2 ;;
        --detach) detach=1; shift ;;
        --wait) wait=1 name=${2:?--wait needs a name}; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "$0: name must be [A-Za-z0-9._-]+: $name" >&2; exit 2; }
[ -n "$wait" ] || [ $# -gt 0 ] || usage

session=ollaya-$name log=ollaya-jobs/$name.log exitf=ollaya-jobs/$name.exit
remote() { ssh -o BatchMode=yes "$host" "$@"; }

if [ -z "$wait" ]; then
    # The three values are quoted for the remote shell, which re-parses ssh's command line.
    remote bash -s "$(printf %q "$name")" "$(printf %q "$dir")" "$(printf %q "$*")" <<'EOF'
set -eu
name=$1 dir=$2 cmd=$3 jobs=$HOME/ollaya-jobs
mkdir -p "$jobs"
rm -f "$jobs/$name.exit"
tmux new-session -d -s "ollaya-$name" "cd $dir && ($cmd) >$jobs/$name.log 2>&1; echo \$? >$jobs/$name.exit"
EOF
    echo "remote-job: started $session on $host, log ~/$log"
    if [ -n "$detach" ]; then
        echo "  attach: ssh $host -t tmux attach -t $session"
        echo "  wait:   $0 $host --wait $name"
        exit 0
    fi
fi

# One ssh per poll: the exit code once the command has ended, otherwise whether the session is alive.
status() {
    remote "if [ -f $exitf ]; then cat $exitf; elif tmux has-session -t '=$session' 2>/dev/null; then echo running; else echo gone; fi" 2>/dev/null || true
}
ssh -n -o BatchMode=yes "$host" "tail -n +1 -F $log 2>/dev/null" &
tailpid=$!
trap 'kill $tailpid 2>/dev/null || true' EXIT
while :; do
    s=$(status)
    case $s in
        running) ;;
        '') echo "remote-job: $host does not answer, still waiting" >&2 ;;
        [0-9]*)
            sleep 2 # let the tail catch up with the last lines
            kill $tailpid 2>/dev/null || true
            echo "remote-job: $name on $host exited $s"
            exit "$s" ;;
        *) echo "remote-job: $session is gone and ~/$exitf was never written (host rebooted?)" >&2; exit 1 ;;
    esac
    sleep "$interval"
done

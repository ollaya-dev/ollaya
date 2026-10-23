# Remote jobs

Which machine does what, and how to send one a job from the Mac without babysitting an ssh session.

## Machines

The ssh aliases are the ones in `~/.ssh/config`; nothing here adds hosts or installs anything. Results name
these machines by hardware (`rtx-4090`, `rtx-5090`, see [Machines](model-lifecycle.md#machines)), never by host.

| Host | Hardware | For | Not for |
|---|---|---|---|
| `choso-wsl` | RTX 4090 (24 GB), WSL2 Ubuntu 24.04 inside the Windows gaming PC "choso" | Training and benches. **Currently reserved for training**: ask before sending anything | Anything while a training run owns the GPU. It sleeps after 60 minutes idle: wake it with `~/bin/choso-uyan` before an ssh |
| `hope-wsl` | RTX 5090, Threadripper 3960X 24C/48T, 64 GB, WSL2 Ubuntu 24.04 on the studio workstation "hope" (shared with a colleague) | CUDA parity, long CPU parity, benches on the 5090 (the site's numbers come from an RTX 5090) | Anything outside tmux: the host reboots unexpectedly, so a job tied to an ssh session is lost with it. Nothing that fills the disk or the GPU while a colleague is working |
| The Mac (M4 Pro) | Apple silicon, local | Development, MLX, CPU parity of small models, packaging | CUDA anything; CPU parity of 2B+ decoders (minutes each, do it on hope) |

## `scripts/remote-job.sh`

Runs a command on a host inside a detached tmux session, streams its log, and exits with the command's exit code.
The command survives a dropped connection or the Mac sleeping; a host reboot ends it, and the script says so.

```sh
# run and wait (polls every 30 s; -i changes that)
scripts/remote-job.sh hope-wsl -n parity-coba -d ollaya/convert -- \
    'python -m ollaya_convert.parity_run --machine rtx-5090 --device cuda coba:0.8b'

# start, come back later
scripts/remote-job.sh hope-wsl -n bench-0.18 --detach -- 'python -m ollaya_convert.bench_run …'
scripts/remote-job.sh hope-wsl --wait bench-0.18
ssh hope-wsl -t tmux attach -t ollaya-bench-0.18
```

- The words after `--` are joined and run by the remote shell from `-d` (relative to the remote home, default the
  home), so quote a pipe or `;` as one word.
- The job is tmux session `ollaya-<name>`; without `-n` the name is a timestamp. Output lands in
  `~/ollaya-jobs/<name>.log` on the remote, the exit code in `~/ollaya-jobs/<name>.exit`. Neither is cleaned up.
- `--wait <name>` resumes waiting on a job, also after the script was interrupted, and returns at once when the
  job has already ended.
- The script never installs anything: tmux and the job's own tools must already be on the host.

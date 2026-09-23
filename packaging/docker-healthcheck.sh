#!/bin/bash
# Docker HEALTHCHECK for the Ollaya image: `GET /` on the local daemon must answer
# "Ollaya is running". Uses bash's /dev/tcp so the image needs no curl or wget.
set -euo pipefail

host=${OLLAYA_HOST:-127.0.0.1:11435}
host=${host#*://}
host=${host%%/*}
port=${host##*:}
[[ $port =~ ^[0-9]+$ ]] || port=11435

exec 3<>"/dev/tcp/127.0.0.1/$port"
printf 'GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
grep -q 'Ollaya is running' <&3

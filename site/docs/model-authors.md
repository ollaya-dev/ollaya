---
title: For model authors: Day-0 support
nav: For model authors
description: Releasing a decision model? Share it with Ollaya before launch and `ollaya run` works on announcement day, with parity against your code and benchmark numbers you can quote.
order: 11
---

# For model authors

Releasing an open decision model? Share it before you announce, and on launch day your announcement can say `ollaya run <your-model>`: one command on Linux, macOS and Windows, on a CPU or an NVIDIA GPU, through the CLI, the HTTP API, the TypeSafe SDK and MCP.

## What you get on launch day

- **Your model in the library**, with a page that credits you and links your repository, and the "Run on Ollaya" badge for your model card.
- **Parity with your code.** Ollaya's runtime is checked against your own inference code before it ships: the same inputs to the model, the same decision on every test question, and the logits within a tolerance of 1e-3 or tighter. Each model page reports the measured difference.
- **Numbers you can quote:** accuracy and calibration on [typed decisions](/results) and Bespoke Labs' public benchmark, and latency on an RTX 5090 and a CPU, measured the same way as every other model on [Ollaya Bench](/results). Each number comes with its raw data.
- **A joint announcement**, if you want one: your post, ours, and a quote from Ollaya's maintainer.

## What we need

- **Read access to the weights** on Hugging Face, in a private or gated repository until launch. Ollaya downloads them from your repository, pinned to a commit and checked by sha256, and never re-hosts them, so they stay yours.
- **Your inference code**: the code that builds the prompt or the rows and reads the answer, and the calibration temperature if you fitted one. A model card alone isn't enough to check parity against.
- **A license** that allows serving the model and publishing a graph derived from it, such as Apache-2.0 or MIT.
- **The announcement date**, with its time zone.

Today the parity check runs the reference in fp32 on a 24 GB GPU, so a decoder of up to about 9B parameters fits. For a larger model, ask first.

## Timeline

A new checkpoint of an architecture Ollaya already runs takes a few days. A new architecture takes longer, since its layout and engine are ported from your code, so share it two weeks before launch if you can.

Until you announce, the work stays private: no public branch, pull request, registry entry or benchmark row. On the day, the model ships in a release, and the site and the bench go live with it.

## Get in touch

Message [@mertcobanov](https://x.com/mertcobanov) on X with the model, its expected date and how to reach you. For a model that is already public, [open an issue](https://github.com/ollaya-dev/ollaya/issues) instead.

## Badge

![Run on Ollaya](/badge/run-on-ollaya.svg)

For a model that is in the library, add this to the model card, with your model's name:

````markdown
[![Run on Ollaya]({{SITE_ORIGIN}}/badge/run-on-ollaya.svg)]({{SITE_ORIGIN}}/library/<your-model>)

## Run it locally

```shell
curl -fsSL {{SITE_ORIGIN}}/install.sh | sh
ollaya run <your-model> --ask "Is a refund requested?" "I was charged twice, please refund me."
```
````

Every model page on this site shows the snippet with its name filled in. Models on Ollaya Bench also have a score badge, on their bench page.

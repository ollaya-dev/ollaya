---
title: Ollaya Cloud data handling
nav: Cloud data
description: What Ollaya Cloud keeps about your requests, for how long, and where: no request content, only metadata.
order: 25
---

# Data handling

**Ollaya Cloud never stores or logs what you send or what it answers.** A state and its answers exist only in memory while the request is answered; no database, log, trace, or metric holds them.

## What is kept

For each request, Ollaya Cloud keeps metadata only, which is what the **Usage** and **Requests** pages show:

- the time, request ID, key, endpoint, and region;
- the model asked for and the model that answered;
- the status and error code, and the number of questions;
- input tokens, the price, and the cost;
- the total, model, and load times;
- your tags (`X-Ollaya-Tags`) and the SDK name and version it sent.

This metadata is kept for 90 days, then deleted. Credits and debits stay in your organization's ledger. Account data (your name, email, organization, members, and keys' masked forms) is kept while the account exists; **Account** → **Download my data** exports it.

## Keys

Ollaya Cloud stores only a keyed hash of each API key, never the key itself, so nobody can read a key back, Ollaya included. See [Keys](/docs/cloud-keys).

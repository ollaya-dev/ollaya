---
title: Ollaya Cloud models and regions
nav: Cloud models
description: The decision models Ollaya Cloud serves, how to choose one, how you pay for them, and where they run.
order: 24
---

# Models and regions

Ollaya Cloud serves Ollaya's own decision models, the same ones you can run locally:

| Family | Models | Kind |
|---|---|---|
| `coba` | `coba:1b`, `coba:400m`, `coba:400m-ternary`, `coba:150m-ternary`, `coba:68m-ternary` | English encoders, zero-shot or trained in-domain; the ternary models are the fastest |
| `sura` | `sura:2b`, `sura:0.8b` | Small decoders, the most accurate |

`GET https://api.{{SITE_HOST}}/v1/models` lists the models your key may call, in TypeSafe's shape. Name a model exactly, tag included (`coba:400m`, not `coba`). The dashboard's **Models** page shows each model's description, release date, Ollaya Bench score, and price, and what the last 30 days of your traffic would have cost on the others.

## Prices

You pay per input token, at the price of the model that answered, in force when the request arrived; output tokens are free, since a decision model answers in one forward pass. The **Models** page shows the current prices. A new price applies from a set time, never to requests already answered. Only `2xx` answers cost anything.

## Model access

An organization can allow every listed model (the default), every model except some, or only some; a key can narrow that further. A model you may not call answers `403 MODEL_NOT_ALLOWED`.

## Regions

Every request is answered in one region. Ollaya Cloud starts in Turkey (`tr`). An organization with a residency region set on the **Settings** page is answered only there, even when that region has no free replica (then it gets `503 QUEUE_FULL`). Any other organization is answered in the region its request reaches, or in another one when that region has no replica of the model.

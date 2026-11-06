---
title: Ollaya Cloud API keys
nav: Cloud keys
description: Authenticate to Ollaya Cloud with API keys, and limit what each key may do, spend, and call.
order: 21
---

# API keys

Every request to `https://api.{{SITE_HOST}}` carries a key in `Authorization: Bearer <key>`. Keys belong to an organization; anyone in it with the Owner, Admin, or Developer role can create them on the dashboard's **Keys** page or through the management API.

A key is `oly_live_` (or `oly_test_` for a test key) followed by 46 characters, the last six of them a checksum, so a mistyped key is refused before it is looked up. Ollaya Cloud stores only a keyed hash of each key and a masked form (`oly_live_…abcd`), so a key is shown once, when it is created, and never again. If you lose it, create a new one.

## What a key may do

| Setting | Effect |
|---|---|
| Scope | `inference` calls models (`/v1/*`, `/api/decide`); `management` manages the organization through the management API (Owners and Admins only). A `read` scope is reserved for read-only access and opens no route yet. |
| Allowed models | Only these models; any other answers `403 MODEL_NOT_ALLOWED` |
| Spend cap | A budget per month or in total; once it is used up the key answers `402 SPEND_CAP_REACHED` |
| Rate and concurrency | Lower limits than the organization's, for one noisy client |
| IP allowlist | Up to 32 addresses or CIDR ranges; calls from elsewhere answer `403 IP_NOT_ALLOWED` |
| Expiry | 30, 90, or 365 days, or none |

The organization's own settings apply on top: its model access list, its budget, and its residency region.

## Keep keys on the server

A key belongs in your server's environment or secret store, never in a web page or a mobile app. Ollaya Cloud refuses any request with an `Origin` header (`403 BROWSER_NOT_ALLOWED`), so a key pasted into front-end code fails at once instead of leaking quietly.

## Revoke a key

Revoke a key on the **Keys** page. Every gateway refuses it within a second, everywhere. The **Audit** page records who created, changed, and revoked each key.

# Third-party notice

The skills in this directory come from **agent-skills** by Addy Osmani. That covers every `*/SKILL.md`, `security-and-hardening/references/hardening-patterns.md` and `_references/*.md`.

- Source: https://github.com/addyosmani/agent-skills
- Commit: `bcab6a1b8503100e8618c3b4e32cc78de43de769`
- License: MIT (full text below)

`THIRD_PARTY_NOTICE.md` and `PROJECT_NOTES.md` are Ollaya's own files, not part of the upstream work.

## Installed from upstream

- Skills: api-and-interface-design, ci-cd-and-automation, code-review-and-quality, debugging-and-error-recovery, documentation-and-adrs, frontend-ui-engineering, git-workflow-and-versioning, incremental-implementation, observability-and-instrumentation, performance-optimization, planning-and-task-breakdown, security-and-hardening (with `security-and-hardening/references/hardening-patterns.md`), shipping-and-launch, spec-driven-development, test-driven-development.
- Shared checklists: upstream `references/` is installed as `_references/`: accessibility-checklist, definition-of-done, observability-checklist, performance-checklist, security-checklist, testing-patterns.

No upstream hooks, agents, commands, plugin manifests or MCP configuration are installed.

## Changes from upstream

The content is unmodified except for the changes below. Where a whole section was removed, the file says so in place.

1. `test-driven-development/SKILL.md`: removed the "**Related:**" line and the body of "Browser Testing with DevTools". Both depend on the Chrome DevTools MCP server, which is not configured here. Its setup lives in the excluded browser-testing-with-devtools skill and adds an MCP server with `npx -y chrome-devtools-mcp@latest`.
2. `ci-cd-and-automation/SKILL.md`: removed the body of "Feeding CI Failures Back to Agents". It described an agent that fixes, commits and pushes on its own after a CI failure.
3. `git-workflow-and-versioning/SKILL.md`:
   - Removed the sentence that recommends `git reset --hard HEAD` when "an agent goes off the rails". That command discards uncommitted work.
   - Removed the `git push origin v1.4.0` line from the release-tag example. Pushing a tag publishes a release.
4. `shipping-and-launch/SKILL.md`: the rollback step `git revert <commit> && git push` is now `git revert <commit>`. Pushing is left to the user.
5. Link fixes, so paths resolve at their new location:
   - `../../references/` became `../_references/` in every `SKILL.md`.
   - `../../../references/` became `../../_references/` in `security-and-hardening/references/hardening-patterns.md`.
   - In `spec-driven-development`, `skills/<name>/SKILL.md` became `../<name>/SKILL.md`. The path to `context-engineering`, which is not installed, was dropped; the name stays.

## MIT License

```
MIT License

Copyright (c) 2025 Addy Osmani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

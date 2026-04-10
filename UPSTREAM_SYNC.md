# Upstream sync notes

This workspace intentionally supports **pulling updates from upstreams** without making accidental upstream pushes easy.

## Outer repo: autoresearch-macos-tomx

Local path:
- `/Users/stephenbeale/Projects/autoresearch-macos-tomx`

Remotes:
- `origin` = `https://github.com/chaosste/autoresearch-macos.git`
- `upstream` = `https://github.com/miolini/autoresearch-macos.git`

Safety:
- default push target is your fork remote: `origin`
- push to `upstream` is disabled via `pushurl=DISABLED`

Typical update flow from upstream master:

```bash
git fetch upstream master
git switch tomx
# inspect differences first
git log --oneline --left-right tomx...upstream/master
# then merge or rebase intentionally
git merge upstream/master
# or: git rebase upstream/master
```

## Inner repo: vendored oh-my-codex source

Local path:
- `/Users/stephenbeale/Projects/autoresearch-macos-tomx/oh-my-codex`

Remote:
- `origin` = `https://github.com/Yeachan-Heo/oh-my-codex.git`

Safety:
- fetch/pull from upstream is enabled
- push back to `Yeachan-Heo/oh-my-codex` is disabled via `pushurl=DISABLED`

Typical update flow from upstream main:

```bash
cd oh-my-codex
git fetch origin main
# inspect first
git log --oneline --left-right main...origin/main
# then update intentionally
git merge origin/main
# or: git rebase origin/main
npm install
npm run build
cd ..
./omx setup --scope project --force --verbose
```

## Verification

Check current remote wiring:

```bash
git remote -v
cd oh-my-codex && git remote -v
```

Expected safe behavior:
- outer repo can still push to your fork (`origin`)
- outer repo cannot push to `miolini/autoresearch-macos`
- inner repo can fetch from `Yeachan-Heo/oh-my-codex`
- inner repo cannot push to `Yeachan-Heo/oh-my-codex`

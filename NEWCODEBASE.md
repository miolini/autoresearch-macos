# Research Orchestration Codebase Architecture

Top-line: yes, it is worth creating a new codebase/layer for this, and yes, I can help structure what should go into it.

## Why a new terminal/session

Because this is a different task shape:

- less "one more train.py patch"
- more productizing the research workflow
- likely involves:
  - extracting reusable prompts/policies
  - deciding boundaries between repo code vs orchestration code
  - maybe designing a portable bundle for other LLMs / Modal / long runs

A fresh session will keep the context cleaner.

## Is it worth a new codebase?

Probably yes, if your goal is any of these:

- reuse the setup across multiple research repos
- run with different LLM backends
- support long unattended runs
- separate "research benchmark code" from "research orchestration system"
- make the prompt/agent/policy layer portable

Right now, the useful pieces are partly entangled across:

- train.py
- OMX_TASK.md
- repo-local agents
- review docs / triage rules
- implicit operational habits from this thread

That's workable for active exploration, but not ideal for:

- portability
- reproducibility
- backend swapping
- long-run automation

## What the new codebase would likely contain

At a high level, I'd expect a reusable "research-orchestration pack" to include:

- task policy
  - OMX_TASK.md-style task contract
- agents
  - the repo-local .codex/agents/*.toml
- research policy docs
  - triage / promotion / failure-family rules
- experiment templates
  - run naming conventions
  - seed ladders
  - output folder contracts
- evaluation conventions
  - what counts as keep / discard / promote
- backend adapters
  - Codex now, maybe other LLMs later
- long-run runner glue
  - local, Modal, possibly queue/batch later

## Why that helps other LLMs

Because the portable value is mostly in:

- the task framing
- the policy logic
- the agent role definitions
- the artifact expectations

not in Codex alone.

So yes:

- the md / yaml / py style package is exactly the right kind of thing if you want portability across:
  - other LLMs
  - long-run infra
  - different experiment repos

## My recommendation

I would not just keep accreting this into the current repo forever.

I would instead create a separate layer/codebase, something like:

- a reusable research-orchestration toolkit
- with repo-specific overlays for each benchmark/project

## Short answer

- New terminal/session? yes
- Worth a new codebase? yes
- Portable beyond Codex? very likely yes
- Should we start by planning architecture rather than coding immediately? definitely yes

If you want, in the next fresh session I'd recommend we do:

1. define the scope of this reusable package
2. separate reusable vs repo-specific elements
3. design the file layout before writing anything

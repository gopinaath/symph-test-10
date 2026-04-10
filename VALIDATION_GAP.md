# Symphony Validation Gap Analysis

## Current State

Symphony manages **which issues to work on** (polling, dispatch, retry, concurrency) but does NOT manage **how to verify the work is correct**.

### What Symphony does today

```
Issue → Dispatch to Agent → Agent generates code (up to max_turns) → Agent exits → Done
                                                                          ↓ (crash)
                                                                    Retry with backoff
```

- Agent completion is trusted at face value — if the agent stops, Symphony marks it complete
- No programmatic verification of correctness
- No test execution after code generation
- No error feedback loop (failing tests → agent retry with context)
- No sub-task decomposition for complex issues
- No completion criteria beyond "agent exited normally"

### What the autonomous-coding quickstart does (that Symphony doesn't)

| Capability | autonomous-coding | Symphony |
|-----------|-------------------|----------|
| Feature decomposition | `feature_list.json` with 200 test cases | None |
| Test execution | Agent runs tests, checks pass/fail | None |
| Red-green loop | Generate → test → if fail, retry with errors | None |
| Completion criteria | All tests pass + 2 consecutive verifications | Agent exits |
| Progress persistence | `feature_list.json` survives session restarts | Issue state only |
| Sub-task tracking | Per-feature pass/fail with JSON | Per-issue only |
| Two-agent pattern | Initializer (plan) + Coder (execute) | Single agent |

### Impact

Without validation, Symphony is a **dispatcher without quality gates**. It can farm out 10 issues in parallel, but has no way to know if the generated code:
- Compiles
- Passes existing tests
- Doesn't break other features
- Actually addresses the issue
- Meets code quality standards

### What needs to be added

```
Issue → Decompose into tasks → For each task:
         Agent generates code → Run validation suite → Pass? → Next task
                                     ↓ Fail
                               Feed errors back → Agent fixes → Re-validate
                                     (up to max_attempts per task)

All tasks complete → Run full test suite → Pass? → Create PR → Done
                                              ↓ Fail
                                         Mark for human review
```

Key components needed:
1. **Validation hooks** — configurable test commands per workspace
2. **Error feedback loop** — pipe test output back as agent context
3. **Completion gates** — programmatic success criteria
4. **Task decomposition** — break issues into verifiable sub-tasks
5. **Progress tracking** — per-task state within an issue
6. **Session persistence** — survive agent restarts without losing progress

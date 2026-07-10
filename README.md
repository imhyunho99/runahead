# runahead

Speculative parallel execution for coding agents.

A coding agent finishes a task and stops to ask what's next. You aren't there. Nothing happens until you come back and answer. `runahead` guesses the answers, runs them in isolated worktrees while you're away, and hands you a queue to pick from.

[Design doc (Korean)](docs/specs/2026-07-10-runahead-design.md)

---

## What makes this different from bypass mode

`--dangerously-skip-permissions` also proceeds without you. The difference is what happens to your choice.

| | bypass mode | runahead |
|---|---|---|
| timelines | one | several |
| your choice | **removed** | **deferred** |
| on a wrong guess | undo it | don't pick it |
| irreversible actions | possible | structurally impossible |

Bypass mode trades control for autonomy. runahead keeps both: the decision still belongs to you, it just happens after the work instead of before it.

## How it works

Speculation is not branch prediction. Nothing in the machine resolves the branch — a slow oracle does, and the oracle is you. So this is run-ahead execution, and its value scales with how long you're away, not with how fast the agent is.

Each guess is an **action**: one prompt, one isolated worktree, one patch. Actions are the unit of storage, acceptance, and learning. That is why a rebase conflict can kill one of them without touching the rest.

Actions fall into two lanes:

- **Fixed lane** — orthogonal work (tests, lint, commit draft). Independent checkboxes. Accepting all of them is the default, so it costs you nothing to review.
- **Predicted lane** — competing work (edge cases, error handling, UI). Radio buttons. You have to read and choose.

The whole thing rests on one constraint:

> **Reviewing N results must cost less than doing one yourself.**

Otherwise runahead hasn't removed work, it has multiplied it. Every design decision below follows from that.

## What it learns

One Beta counter per `(task kind, action kind)`. That's it — no fine-tuning, no embeddings, no vector database. A session yields three to five labels; there is no gradient to see.

```
feature|write-tests      alpha=47 beta=6    p=0.87
feature|add-edge-cases   alpha=12 beta=19   p=0.39
feature|responsive-ui    alpha=2  beta=1    p=0.67  <- three tries. not trusted.
```

Confidence `p` turns three dials at once: how many competing variants to generate, whether the action earns children, and whether it can be auto-accepted. Auto-accept requires a high mean **and** a tight posterior — 2 of 3 successes has a mean of 0.67 and tells you nothing.

Three consequences fall out of using Beta rather than a ratio:

**Exploration is built in.** Candidates are ranked by sampling from the posterior (Thompson sampling), not by its mean. Without this, the system dies of exposure bias: it proposes A, you accept the A in front of you, the statistics tilt toward A, and B — which you actually wanted — never appears on screen to be chosen. A wide posterior occasionally draws high, and that is the only way the truth gets a chance to surface. There is a regression test for exactly this.

**Graduation and demotion are free.** As `p` rises an action stops being offered as one of three and starts being applied silently. As `p` falls it drops back. Your personal `/ship` grows on its own instead of being written by hand — that is the actual delta over a fixed post-task script.

**The queue gets shorter, not deeper.** Reversibility caps depth long before confidence does. What improves with learning is that you're asked less.

Priors are hierarchical: a global habit seeds each new repo, so cold start happens once rather than once per repository.

### The signal that matters

Accept rate is measured only over what the system chose to show you. It climbs as the system narrows onto its own habits, which looks like learning and isn't.

**Miss rate** — how often you ignored the queue entirely and asked for something else — cannot be gamed that way. It names the actions the predictor failed to imagine. Depth is gated on it.

Rebase conflicts are the third label. Two actions believed orthogonal whose patches don't compose is not a bug; it's `do not propose this pair together again`.

## The boundary

Speculation is justified exactly where rollback is free. Inside a worktree it is. `git push` is not, nor is a deploy, a migration, or an outbound POST.

runahead never crosses that line, and confidence never unlocks it. At `p = 0.99` it still does not push. The boundary is drawn by the machine, not by checkpoints you place by hand — otherwise you couldn't walk away, which was the entire point.

A budget (tokens, wall clock, action count) bounds the reversible work too. Nobody is watching for thirty minutes; without a ceiling the predicted lane will happily inflate itself.

## Storage

```
~/.runahead/         yours. permanent. never committed, never uploaded.
  priors.json        global Beta counters
  repos/<id>.json    per-repo counters, seeded from the global prior
  history.jsonl      accepts, rejects, misses, conflicts

~/.cache/runahead/   where speculation actually runs
  worktrees/<repo>/<action>

.git/runahead/       this repo's. disposable.
  patches/  queue.json
```

Learning data stays out of the repo on purpose. Commit it and your habits average with your teammates', and an averaged habit predicts nobody.

Worktrees live outside the repository, and that is load-bearing. Hand a coding agent a cwd inside `.git` and it edits nothing, returns success, and bills you for the tokens — the empty patch is the only symptom. Put them in the working tree instead and `git status` goes dirty, which is precisely what `accept` refuses to run against.

## Install

Requires Python 3.10+, git, and a coding agent CLI on `PATH`. No dependencies.

```bash
git clone https://github.com/imhyunho99/runahead
cd runahead && pip install -e .
```

## Use

Commit the task you just finished, then walk away.

```bash
runahead run "add retry logic to the http client"

# when you come back
runahead queue
runahead accept add-edge-cases error-handling
```

If the queue offered nothing you wanted, say so. This is the most valuable thing you can tell it:

```bash
runahead miss "write a migration for the new column"
runahead stats
```

```
task (feature): add retry logic to the http client

auto-accepted (graduated)
  [x] write-tests        p=0.98 (n=42)  write-tests -> run-tests -> draft-commit

needs review
  [ ] add-edge-cases     p=0.71 (n=31)  add-edge-cases
  ( ) error-handling#1   p=0.44 (n=18)  error-handling
  ( ) error-handling#2   p=0.44 (n=18)  error-handling
  [ ] responsive-ui      p=0.31 (n=4)   responsive-ui

stopped at the reversibility boundary: push, pr, deploy
budget: tokens 84,000/200,000 · 18m/30m · actions 6/12
```

You answer per line. Rebase does the composing. You never see a combination matrix — that would multiply the cost this tool exists to divide.

## Agents

The core knows nothing about Claude. One seam:

```python
run(action, worktree) -> Result
```

| adapter | status |
|---|---|
| `claude` | working |
| `conductor` | planned |
| `codex` | planned |

## Relation to conductor

[conductor](https://github.com/j0j1j2/claude-conductor) throws one task at several agents and picks a winner. It parallelizes **across space, at one moment in time**.

runahead parallelizes **the same trunk, across time**.

> runahead decides *what to do next*. conductor can decide *how to do it*.

They compose: an action with low confidence needs competing variants, which is exactly what conductor is good at. runahead works without it, and gets a stronger predicted lane with it.

## Status

v1, exercised end to end against a real `claude`: it read a freshly committed `parse_duration()` that silently returned `0` on garbage input, proposed error handling and documentation, wrote validation plus tests for the first, and — since both patches touched the same file — applied one, isolated the other as a conflict, rolled the tree back clean, and stored the pair as a label.

The core has 35 tests and no LLM in any of them. Swap in `FakeExecutor` and the scheduler, budget, boundary, and patch merge all run for real. That's the second reason the executor is a separate seam.

Chains run one at a time. The name says parallel; v1 is serial on purpose, because the rate limit is the real bottleneck and the safe concurrency has to be measured before it is chosen.

The unproven claim is whether `p` converges to something useful at three to five samples per session. Falling miss rate over many sessions is the only evidence that would settle it. If it doesn't converge, runahead collapses into a hand-written post-task script, and the interesting part was never real.

MIT.

---
---

# runahead (한국어)

코딩 에이전트를 위한 투기적 병렬 실행.

에이전트는 작업을 끝내면 멈춰서 묻는다. "다음에 뭘 할까요?" 당신은 자리에 없다. 돌아와 답할 때까지 아무 일도 일어나지 않는다. `runahead`는 그 답을 미리 추측해서, 자리를 비운 동안 격리된 worktree에서 실행해두고, 돌아온 당신에게 고를 큐를 내민다.

[설계 문서](docs/specs/2026-07-10-runahead-design.md)

## 바이패스 모드와 무엇이 다른가

`--dangerously-skip-permissions`도 사람 없이 진행한다. 차이는 **당신의 선택이 어떻게 되는가**에 있다.

| | 바이패스 모드 | runahead |
|---|---|---|
| 타임라인 | 하나 | 여러 갈래 |
| 당신의 선택 | **제거됨** | **연기됨** |
| 추측이 틀리면 | 되돌려야 함 | 안 고르면 그만 |
| 비가역 행동 | 가능함 | 구조적으로 불가능 |

바이패스 모드는 통제권을 팔아 자율성을 산다. runahead는 둘 다 가진다. 결정권은 여전히 당신 것이고, 다만 작업 **이전**이 아니라 **이후**에 행사될 뿐이다.

## 어떻게 동작하는가

이것은 분기 예측이 아니다. 분기를 해소하는 주체가 기계 안에 없기 때문이다. 해소하는 것은 **느린 오라클, 즉 사람**이다. 그래서 이건 run-ahead execution이고, 그 가치는 에이전트의 속도가 아니라 **당신이 자리를 비운 시간에 비례**한다.

추측 하나가 곧 **행동(action)**이다. 프롬프트 하나, 격리된 worktree 하나, 패치 하나. 행동이 저장·수락·학습의 단위다. rebase 충돌이 다른 것들을 건드리지 않고 그 하나만 죽일 수 있는 이유가 여기 있다.

행동은 두 레인으로 갈린다.

- **고정 레인** — 직교적 작업(테스트, 린트, 커밋 초안). 독립적인 체크박스. 전부 수락이 기본값이라 검토 비용이 사실상 없다.
- **예측 레인** — 경쟁적 작업(엣지케이스, 에러 핸들링, UI). 라디오 버튼. 읽고 골라야 한다.

전체가 단 하나의 제약 위에 서 있다.

> **N개를 훑고 고르는 비용이 직접 1개를 하는 비용보다 작아야 한다.**

그렇지 않으면 runahead는 일을 덜어준 게 아니라 **N배로 늘린 것**이다. 아래의 모든 결정이 여기서 나온다.

## 무엇을 학습하는가

`(작업 유형, 행동 유형)` 쌍마다 Beta 카운터 하나. 그게 전부다. 파인튜닝도, 임베딩도, 벡터 DB도 없다. 세션당 라벨이 서너 개뿐이라 볼 그레이디언트가 애초에 없다.

```
feature|write-tests      alpha=47 beta=6    p=0.87
feature|add-edge-cases   alpha=12 beta=19   p=0.39
feature|responsive-ui    alpha=2  beta=1    p=0.67  <- 시도 3번. 못 믿는다.
```

확신도 `p` 하나가 다이얼 세 개를 동시에 돌린다. 경쟁 변형을 몇 개 만들지, 이 행동이 자식을 낳을 자격이 있는지, 자동 수락해도 되는지. 자동 수락에는 높은 평균**과 함께** 좁은 사후분포가 필요하다. 3번 중 2번 성공은 평균 0.67이지만 아무것도 말해주지 않는다.

비율 대신 Beta를 쓰면 세 가지가 딸려 나온다.

**탐색이 원리적으로 내장된다.** 후보를 사후분포의 평균이 아니라 **사후분포에서 샘플링**해서 순위를 매긴다(Thompson sampling). 이게 없으면 시스템은 노출 편향으로 죽는다. A를 제안하고 → 눈앞의 A를 당신이 수락하고 → 통계가 A로 기울고 → 정작 당신이 원하던 B는 **화면에 뜬 적이 없어 선택될 기회조차 없다.** 넓은 사후분포는 가끔 높은 값을 뽑고, 그것이 진실이 드러날 유일한 통로다. 정확히 이걸 지키는 회귀 테스트가 있다.

**졸업과 강등이 공짜다.** `p`가 오르면 그 행동은 셋 중 하나로 제시되기를 그만두고 조용히 적용되기 시작한다. `p`가 떨어지면 도로 내려온다. 당신만의 `/ship`이 손으로 쓰이는 대신 **저절로 자란다.** 이것이 손으로 쓴 후처리 스크립트 대비 실제 델타다.

**큐는 깊어지는 게 아니라 짧아진다.** 확신도보다 가역성이 먼저 깊이의 천장을 친다. 학습으로 나아지는 것은 **당신이 덜 질문받는다**는 것이다.

사전분포는 계층적이다. 전역 습관이 새 레포의 출발점을 깔아주므로, 콜드스타트는 레포마다가 아니라 딱 한 번 일어난다.

### 진짜 지표

수락률은 **시스템이 보여주기로 고른 것들 중에서만** 측정된다. 시스템이 자기 습관으로 좁혀 들어갈수록 올라간다. 학습처럼 보이지만 아니다.

**miss rate** — 큐를 통째로 무시하고 전혀 다른 걸 시킨 비율 — 는 그런 식으로 조작되지 않는다. 예측기가 **상상하지 못한** 행동이 무엇인지 알려준다. 깊이는 여기에 걸어둔다.

rebase 충돌이 세 번째 라벨이다. 직교하리라 믿었던 두 행동의 패치가 합쳐지지 않는 것은 버그가 아니라 `이 둘을 같이 제안하지 마라`는 뜻이다.

## 경계

투기가 정당한 영역은 **롤백이 공짜인 영역과 정확히 일치한다.** worktree 안은 그렇다. `git push`는 아니다. 배포도, 마이그레이션도, 외부로 나가는 POST도 아니다.

runahead는 그 선을 넘지 않으며, **확신도는 이 잠금을 절대 풀지 못한다.** `p = 0.99`여도 push하지 않는다. 경계는 사람이 손으로 찍는 체크포인트가 아니라 기계가 긋는다. 안 그러면 자리를 비울 수 없고, 그게 애초에 이 도구의 전부였다.

예산(토큰, 벽시계, 행동 수)이 가역적인 작업에도 상한을 건다. 30분 동안 아무도 보고 있지 않다. 천장이 없으면 예측 레인은 기꺼이 스스로를 부풀린다.

## 저장

```
~/.runahead/        당신 것. 영구. 커밋되지 않고 업로드되지 않는다.
  priors.json       전역 Beta 카운터
  repos/<id>.json   레포별 카운터. 전역을 prior로 삼아 출발한다.
  history.jsonl     수락 / 거부 / miss / 충돌

.git/runahead/      이 레포 것. 일회용.
  worktrees/  patches/  queue.json
```

학습 데이터를 레포 밖에 두는 것은 의도다. 커밋하면 당신의 습관이 동료의 습관과 평균되고, **평균된 습관은 누구도 예측하지 못한다.**

## 설치

Python 3.10+, git, 그리고 `PATH` 위의 코딩 에이전트 CLI. 의존성 없음.

```bash
git clone https://github.com/imhyunho99/runahead
cd runahead && pip install -e .
```

## 사용

방금 끝낸 작업을 커밋하고, 자리를 뜬다.

```bash
runahead run "http 클라이언트에 재시도 로직 추가"

# 돌아와서
runahead queue
runahead accept add-edge-cases error-handling
```

큐에 원하는 게 하나도 없었다면 그렇게 말하라. **당신이 줄 수 있는 가장 값진 정보다.**

```bash
runahead miss "새 컬럼 마이그레이션 작성"
runahead stats
```

당신은 줄 단위로 예/아니오만 한다. 조합은 rebase가 만든다. 조합표는 절대 보지 않는다 — 그건 이 도구가 나누려는 바로 그 비용을 곱하는 짓이다.

## 에이전트

코어는 Claude가 무엇인지 모른다. 이음매는 하나다.

```python
run(action, worktree) -> Result
```

| 어댑터 | 상태 |
|---|---|
| `claude` | 동작함 |
| `conductor` | 예정 |
| `codex` | 예정 |

## conductor와의 관계

[conductor](https://github.com/j0j1j2/claude-conductor)는 하나의 과제를 여러 에이전트에게 던지고 승자를 고른다. **한 시점에서 공간 방향으로** 병렬화한다.

runahead는 **같은 줄기를 시간 방향으로** 병렬화한다.

> runahead는 *다음에 무엇을 할지* 정한다. conductor는 *그것을 어떻게 할지* 정할 수 있다.

둘은 합쳐진다. 확신이 낮은 행동에는 경쟁 변형이 필요하고, 그건 정확히 conductor가 잘하는 일이다. runahead는 conductor 없이도 돌아가고, 있으면 예측 레인이 강해진다.

## 상태

v1. 실제 `claude`로 전 구간을 돌렸다. 방금 커밋된 `parse_duration()`이 잘못된 입력에 조용히 `0`을 반환하는 것을 읽어내, 에러 처리와 문서화를 제안하고, 전자에 검증과 테스트를 붙였다. 두 패치가 같은 파일을 건드렸으므로 하나만 적용하고 나머지는 충돌로 격리한 뒤 트리를 깨끗하게 되돌리고, 그 쌍을 라벨로 저장했다.

코어에 테스트 35개가 있고 그 안에 LLM은 하나도 없다. `FakeExecutor`를 꽂으면 스케줄러·예산·경계·패치 머지가 전부 실제로 돈다. 실행기를 별도 이음매로 분리한 두 번째 이유가 이것이다.

체인은 하나씩 순차로 돈다. 이름은 병렬이지만 v1은 의도적으로 직렬이다. 레이트 리밋이 실제 병목이고, 안전한 동시 실행 수는 고르기 전에 측정해야 하기 때문이다.

또한 `~/.cache/runahead/` 아래에 worktree를 둔다. `.git` 안이면 에이전트가 아무것도 안 하고 성공을 반환한다.

**아직 증명되지 않은 주장**은 세션당 서너 개의 샘플로 `p`가 쓸 만하게 수렴하는가이다. 여러 세션에 걸친 miss rate 하락만이 이를 판정할 수 있다. 수렴하지 않는다면 runahead는 손으로 쓴 후처리 스크립트로 쪼그라들고, 흥미로웠던 부분은 처음부터 없었던 것이 된다.

MIT.

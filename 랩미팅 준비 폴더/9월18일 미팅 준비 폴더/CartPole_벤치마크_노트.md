# PPO CartPole-v1, GPU가 CPU보다 2.5배 느렸던 이유

Lab Note · RL Runner Benchmark

강화학습을 처음 돌려보면서 러너(runner) 구조와 GPU 활용을 실측으로 확인한 기록.
연구실 워크스테이션에서 재현 가능.

- **날짜**: 2026-09-15
- **GPU**: RTX PRO 6000 Blackwell Max-Q (98GB)
- **CPU**: 32 logical cores
- **환경**: conda RL_Book · Python 3.9.25 · torch 2.8.0+cu128

원본 아티팩트(인터랙티브 버전): https://claude.ai/artifact/CFJ2nd5gus5fdqZxnmSLQJ

---

## 01. 배경 — 왜 확인했나

PPO로 CartPole-v1을 학습시키던 중 GPU를 "풀로" 쓰고 있는지 궁금해서 `nvidia-smi dmon`으로 학습 중
사용률을 실측했다. 결과는 예상 밖이었다 — GPU 사용률이 10%도 안 됐고, 같은 학습을 CPU로 돌리니
오히려 더 빨랐다. 아래는 그 과정과 수치, 그리고 연구실 환경에 적용할 결론이다.

## 02. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | RTX PRO 6000 Blackwell Max-Q |
| VRAM | 97,887 MiB |
| CPU | 32 logical cores |
| 실행 환경 | conda env "RL_Book" (Python 3.9.25) |
| 주요 패키지 | torch 2.8.0+cu128 · ray 1.13.0 · gym 0.22.0 · pybullet 3.2.5 |
| 알고리즘 / 환경 | PPO / CartPole-v1 (n_envs=1) |

> 참고: "GPU 환경"과 "서버 환경"은 별도 머신이 아니라 이 워크스테이션 한 대를 가리킨다.
> 이 프로젝트는 반드시 conda 가상환경 `RL_Book`에서 실행해야 한다 (시스템 Python에는
> ray/tensorboard가 없어 `ModuleNotFoundError` 발생).

## 03. Runner 구조 — 3가지 실행 방식

이 프레임워크는 `n_envs`와 `distributed_processing_type` 설정으로 아래 세 러너 중 하나가 선택된다.

| 러너 | 조건 | 특징 |
|---|---|---|
| `Runner` | n_envs = 1 | 환경 1개, 단일 프로세스 순차 실행. 분산 처리 오버헤드 없음. 이번 벤치마크가 사용한 방식. |
| `MultiEnvRunner` | n_envs > 1, sync | Ray로 환경을 여러 프로세스에 분산. 매 라운드마다 **전체** 환경이 끝날 때까지 대기 후 학습 — 가장 느린 환경이 전체를 지연시킴(straggler 문제). |
| `MultiEnvAsyncRunner` | n_envs > 1, async | Ray로 분산하되 **먼저 끝난** 환경 결과부터 즉시 학습에 반영. 처리량은 높지만 정책 스냅샷이 환경마다 살짝 어긋날 수 있음. |

## 04. 실측 결과

### GPU vs CPU — 100,000 스텝 소요 시간

| 모드 | 총 소요 시간 | steps/sec |
|---|---|---|
| GPU (`use_cuda: True`), 1차 | 0:03:27 (207초) | ~483 |
| GPU (`use_cuda: True`), 2차 (재현성 확인) | 0:03:24 (204초) | ~487 |
| **CPU (`use_cuda: False`)** | **0:01:22 (82초)** | **~1,220** |

**→ CPU가 GPU보다 약 2.5배 빠름.**

### 학습 중 GPU 실사용률 (`nvidia-smi dmon`, 240초 샘플링)

- 평균 SM(연산) 사용률: **5.8%**, 최대 **9%**
- VRAM: 707MB / 97,887MB (0.7%)
- 전력: 64~73W / 300W

GPU는 대부분 대기 상태였다.

## 05. 원인

`agents/actor.py`의 `select_action()`이 매 스텝마다 상태 1개(batch=1)를 CPU→GPU로 보내 forward 하고
다시 CPU로 받아온다 (`runner/environment_loop.py`의 while 루프에서 스텝마다 반복). 이 왕복
오버헤드(커널 실행 지연 + PCIe 전송)가 512유닛 MLP 한 번의 실제 연산량보다 커서, GPU는 대부분
다음 호출을 기다리며 놀았다.

## 06. n_envs=4/16 × sync/async × CPU/GPU 전수 비교 (2026-09-16 추가)

이번엔 `MultiEnvRunner`(sync)와 `MultiEnvAsyncRunner`(async)를 CartPole-v1과 AntBulletEnv-v0
양쪽에서, `n_envs=4`와 `n_envs=16`(32코어 중 더 활용) 두 규모로, CPU/GPU 각각 실측했다.

> 참고: 저장소에 있던 `CartPole-v1-4-sync.yaml` 같은 변형 파일은 CLI로 바로 실행이 안 된다
> (`main.py`가 `-e` 인자로 내부 `env_name`을 덮어써서 `gym.error.NameNotFound` 발생). 그래서
> 매번 `CartPole-v1.yaml`/`AntBulletEnv-v0.yaml` 본체를 고쳐 실행 → 측정 → `git checkout`으로
> 원복하는 방식으로 진행했다.

**CartPole-v1 (100,000 스텝)**

| n_envs | 처리방식 | GPU | steps/sec |
|---|---|---|---|
| 1 | - | CPU | 1,220 |
| 1 | - | GPU | 487 |
| 4 | sync | CPU | 1,724 |
| 4 | async | CPU | **2,000** |
| 4 | sync | GPU | 556 |
| 4 | async | GPU | 806 |
| 16 | sync | CPU | 2,000 |
| 16 | async | CPU | 1,923 |
| 16 | sync | GPU | 559 |
| 16 | async | GPU | 794 |

**AntBulletEnv-v0 (20,000 스텝 — 벤치마크용으로 축소, 원래 3,000,000)**

| n_envs | 처리방식 | GPU | steps/sec |
|---|---|---|---|
| 1 | - | CPU | 625 |
| 1 | - | GPU | 323 |
| 4 | sync | CPU | 1,111 |
| 4 | async | CPU | **1,333** |
| 4 | sync | GPU | 500 |
| 4 | async | GPU | 571 |
| 16 | sync | CPU | 870 |
| 16 | async | CPU | **1,333** |
| 16 | sync | GPU | 328 |
| 16 | async | GPU | 455 |

**핵심 발견**

1. **모든 조합에서 CPU가 GPU를 이긴다.** 두 환경, 모든 n_envs, sync/async 무관하게 일관됨.
2. **async는 sync보다 항상 같거나 빠르다.** AntBulletEnv에서 특히 뚜렷함 — `n_envs=16 sync CPU`(870)가
   `n_envs=4 sync CPU`(1,111)보다 오히려 느려진다 (워커가 늘수록 straggler 효과가 커짐). async는
   4→16으로 늘려도 1,333으로 유지 — **async가 스케일에 더 견고함.**
3. **4→16 확장은 이득이 없거나 손해.** 두 환경 모두 CPU async 기준 n_envs=4에서 이미 처리량이
   포화된다. GPU 조합은 16에서 오히려 더 느려지기도 한다 (여러 프로세스가 물리 GPU 1개를 두고
   경합하는 것으로 추정).
4. **이 실험 기준 최적 설정: `use_cuda: False`, `n_envs=4`, `distributed_processing_type: "async"`.**
   두 환경 모두에서 실측상 가장 빠른 지점이었고, 16까지 늘리는 건 이득이 없었다.

## 07. 연구실 환경 적용 가이드

**결론: "GPU를 켰다" ≠ "GPU를 효율적으로 쓴다", "환경을 더 늘렸다" ≠ "더 빨라진다"**

- **가벼운 환경** (CartPole류, 저차원 상태, 작은 MLP) → `use_cuda: False`. `n_envs=1`도 충분히
  빠르지만, `n_envs=4, async`가 실측상 더 빠름(2,000 steps/sec) — 다만 Ray 오버헤드 대비 이득이
  크지 않으니 구현 단순성이 더 중요하면 `n_envs=1`도 무방.
- **무거운 환경** (AntBulletEnv류 물리 시뮬레이션) → 역시 `use_cuda: False` + `n_envs=4, async`가
  실측상 최적. 이미지 관측처럼 네트워크 자체가 커지는 경우에만 GPU 재검토가 필요.
- **병렬 규모는 무조건 늘린다고 좋아지지 않는다.** sync는 워커가 늘수록 느려질 수 있고(straggler),
  async도 특정 지점(이 실험에서는 n_envs=4)에서 이미 포화된다. 늘리기 전에 항상 실측할 것.
- 결정 전에 항상 `nvidia-smi dmon -s pucm -d 1`로 학습 중 SM 사용률을 실측해서 병목이
  CPU(환경)인지 GPU(연산)인지 먼저 확인한다.

## 08. 재현 방법

```bash
# RL_Book conda 환경으로 실행 (시스템 python엔 ray/tensorboard 없음)
"C:/Users/User/WorkSpace/Donghun/miniconda3/envs/RL_Book/python.exe" main.py -a ppo -e CartPole-v1

# 학습 중 GPU 사용률 실측
nvidia-smi dmon -s pucm -d 1 -o T
```

---

*원본 기록: `ENVIRONMENT.md` (프로젝트 루트) — 다음 비교 예정: 이미지 관측 등 더 큰 네트워크에서 GPU가
유리해지는 지점, AntBulletEnv를 원래 스텝 수(3,000,000)로 실제 학습 수렴까지 완주했을 때 소요 시간*

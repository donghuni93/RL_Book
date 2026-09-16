# 실행 환경 (Environment)

이 문서는 이 프로젝트를 실행하는 머신의 스펙과 설정을 기록해서,
학습 중 드는 "이 정도 걸리는 게 맞나?", "GPU를 잘 쓰고 있나?" 같은 질문을
다시 확인할 때 참고하기 위한 것이다.

마지막 확인일: 2026-09-15

## 하드웨어

- OS: Windows 10 (build 26200)
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  - VRAM: 97,887 MiB (~98GB)
  - Driver: 581.80
- CPU: 논리 코어 32개

> 참고: "GPU 환경"과 "서버 환경"은 별도 머신이 아니라 **동일한 이 워크스테이션**을 가리킨다.

## 파이썬 실행 환경 (중요)

이 프로젝트는 **conda 가상환경 `RL_Book`**에서 실행해야 한다.
시스템 PATH의 `python`(3.10, `C:\Users\User\AppData\Local\Programs\Python\Python310`)에는
`ray`, `tensorboard` 등이 설치되어 있지 않아 바로 실행하면 `ModuleNotFoundError`가 난다.

- conda env 이름: `RL_Book`
- 위치: `C:\Users\User\WorkSpace\Donghun\miniconda3\envs\RL_Book`
- Python: 3.9.25
- 실행 예시:
  ```bash
  "C:/Users/User/WorkSpace/Donghun/miniconda3/envs/RL_Book/python.exe" main.py -a ppo -e CartPole-v1
  ```
  또는 `conda activate RL_Book` 후 `python main.py ...`

### 주요 패키지 버전 (RL_Book env)

| 패키지 | 버전 |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| ray | 1.13.0 |
| tensorboard | 2.6.0 |
| gym | 0.22.0 |
| pybullet | 3.2.5 |
| numpy | 1.24.4 |

`torch.cuda.is_available() == True`, GPU 이름 정상 인식 확인됨 (`NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`).

`requirements.txt` / `requirements_windows.txt`에는 `ray[default]==1.13`으로 핀 고정되어 있고,
Windows에서는 cp39(Python 3.9) 휠로 설치하라는 주석이 있음 — `RL_Book` env가 Python 3.9인 이유.

## Runner 관련 설정 (Ray 분산 처리)

`config/agents/*/*.yaml`에서 `n_envs`, `distributed_processing_type`으로 어떤 Runner를 쓸지 결정됨
(자세한 동작 방식은 `docs/runner/*.md`, `runner/*.py` 참고).

- `n_envs: 1` → `Runner` (단일 환경, 분산처리 없음)
- `n_envs > 1`, `distributed_processing_type: "sync"` → `MultiEnvRunner` (Ray 기반 동기 분산)
- `n_envs > 1`, `distributed_processing_type: "async"` → `MultiEnvAsyncRunner` (Ray 기반 비동기 분산)

기본 제공되는 4-env 설정 파일들(`*-4-sync.yaml`, `*-4-async.yaml`)은
`num_cpus: 4, num_gpus: 1`로 고정되어 있음 → 이 머신은 32코어라 28개 코어가 남는다.
더 많은 병렬성을 실험하려면 `num_cpus`/`n_envs`를 올린 변형 config가 필요.

## 실측 벤치마크 로그

여기에 실제로 돌려본 결과를 이어서 기록한다. (steps/sec, 소요 시간 등)

### 2026-09-15: `ppo CartPole-v1` (n_envs=1, `Runner`)

- 실행: `config/agents/ppo/CartPole-v1.yaml` 기본값 (`max_environment_steps: 100000`, `n_steps: 128`)
- 실행 커맨드: `"C:/Users/User/WorkSpace/Donghun/miniconda3/envs/RL_Book/python.exe" main.py -a ppo -e CartPole-v1`
- 결과 디렉터리: `results/models/ppo_CartPole-v1_2026-09-15_14-17-20/`
- Start: 2026-09-15 14:17:23 / End: 2026-09-15 14:20:50
- **총 소요 시간: 0:03:27 (207초) → 약 483 steps/sec**
- 학습은 정상적으로 수렴 (100 에피소드 근방부터 `returns_mean` 500 도달, 이후 대부분 유지)
- GPU 사용률은 실험 전 기준 idle(0%)이었고, CartPole은 env.step 자체가 가벼워 GPU가 병목이 아닐 것으로 예상됨 (다음에 `nvidia-smi dmon`으로 학습 중 실측 예정)

### 2026-09-15: GPU 실사용률 측정 (`nvidia-smi dmon`, 학습 중 240초 샘플링)

- 평균 SM(연산) 사용률: **5.8%**, 최대 9%
- VRAM 707MB / 97,887MB (0.7%), 전력 64~73W / 300W
- 원인: `agents/actor.py:126`에서 매 스텝마다 상태 1개(batch=1)를 CPU→GPU로 보내 forward 후 다시 CPU로 받아옴
  (`runner/environment_loop.py:121`). 이 왕복 오버헤드가 512유닛 MLP 1회 연산량보다 커서 GPU가 대부분 대기.

### 2026-09-15: GPU vs CPU 비교 (`ppo CartPole-v1`, n_envs=1, 100,000 스텝 동일 조건)

`config/agents/ppo/CartPole-v1.yaml`의 `use_cuda`만 바꿔서 비교 (테스트 후 `True`로 원복 완료).

| 모드 | 총 소요시간 | steps/sec |
|---|---|---|
| GPU (`use_cuda: True`), 1차 | 0:03:27 (207초) | ~483 |
| GPU (`use_cuda: True`), 2차 (재현성 확인) | 0:03:24 (204초) | ~487 |
| **CPU (`use_cuda: False`)** | **0:01:22 (82초)** | **~1,220** |

**CPU가 GPU보다 약 2.5배 빠름.** CartPole처럼 상태가 작고(float 4개) 네트워크도 작은(hidden 512, 1층) 환경에서는
스텝마다 발생하는 GPU 왕복 오버헤드(커널 실행 지연 + PCIe 전송)가 실제 연산 이득보다 커서,
GPU를 쓰는 게 오히려 손해라는 게 실측으로 확인됨.

### 연구실 환경 설정 권장 (2026-09-15 기준 결론)

- **가벼운 환경(CartPole, 작은 MLP, 상태/행동이 저차원)**: `use_cuda: False`, `n_envs: 1` (`Runner`).
  GPU 자원을 아예 안 쓰는 게 더 빠르고, 공용 GPU 서버 자원도 아낄 수 있음.
- **무거운 환경(AntBulletEnv처럼 물리 연산이 크거나, 이미지 기반 관측처럼 네트워크가 큰 경우)**: GPU 사용 + 필요시
  `n_envs > 1` (`MultiEnvRunner`/`MultiEnvAsyncRunner`)로 CPU 코어를 여러 개 써서 env.step() 병목을 분산하는 게 유효.
  이 경우도 `nvidia-smi dmon`으로 실측 후 판단할 것 (짐작으로 GPU를 켜지 말 것).
- 원칙: **"GPU를 켰다"와 "GPU를 효율적으로 쓴다"는 다른 문제.** 항상 `nvidia-smi dmon -s pucm -d 1`로 학습 중 SM 사용률을 재보고 병목이 CPU(env)인지 GPU(네트워크 연산)인지 먼저 확인한 뒤 설정을 결정할 것.

### 2026-09-16: n_envs=4/16 × sync/async × CPU/GPU 전수 비교

`config/agents/ppo/*-4-sync.yaml` 등 저장된 변형 파일은 **CLI로 직접 실행 불가**하다는 걸 먼저 확인함
(`main.py`가 `-e` 인자로 `config['env_name']`을 덮어써서, 파일 이름과 실제 gym env id가 달라지면
`gym.error.NameNotFound` 발생). 그래서 매 조합마다 `CartPole-v1.yaml`/`AntBulletEnv-v0.yaml` 본체를
직접 고쳐서 실행 → 측정 → 원복(`git checkout HEAD --`)하는 방식으로 테스트함.

**CartPole-v1 (100,000 스텝 기준)**

| n_envs | 처리방식 | GPU | 소요시간 | steps/sec |
|---|---|---|---|---|
| 1 | - | CPU | 82s | 1,220 |
| 1 | - | GPU | 205s | 487 |
| 4 | sync | CPU | 58s | 1,724 |
| 4 | async | CPU | 50s | **2,000** |
| 4 | sync | GPU | 180s | 556 |
| 4 | async | GPU | 124s | 806 |
| 16 | sync | CPU | 50s | 2,000 |
| 16 | async | CPU | 52s | 1,923 |
| 16 | sync | GPU | 179s | 559 |
| 16 | async | GPU | 126s | 794 |

**AntBulletEnv-v0 (20,000 스텝 기준, 벤치마크용으로 축소 — 원래 3,000,000)**

| n_envs | 처리방식 | GPU | 소요시간 | steps/sec |
|---|---|---|---|---|
| 1 | - | CPU | 32s | 625 |
| 1 | - | GPU | 62s | 323 |
| 4 | sync | CPU | 18s | 1,111 |
| 4 | async | CPU | 15s | **1,333** |
| 4 | sync | GPU | 40s | 500 |
| 4 | async | GPU | 35s | 571 |
| 16 | sync | CPU | 23s | 870 |
| 16 | async | CPU | 15s | **1,333** |
| 16 | sync | GPU | 61s | 328 |
| 16 | async | GPU | 44s | 455 |

**핵심 발견**

1. **모든 조합에서 CPU가 GPU를 이김.** 두 환경, 모든 n_envs, sync/async 무관하게 CPU가 항상 더 빠름 —
   두 네트워크(512유닛 1층 / 64-64-64 3층) 모두 GPU가 부담을 상쇄할 만큼 크지 않기 때문.
2. **async는 sync보다 항상 같거나 빠름.** 특히 AntBulletEnv에서 차이가 뚜렷함 —
   `n_envs=16, sync, CPU`(870)가 오히려 `n_envs=4, sync, CPU`(1,111)보다 느려짐 (스트래글러 효과가
   워커 수에 비례해 커짐). async는 4→16으로 늘려도 1,333으로 동일하게 유지됨 — **async가 스케일에
   더 견고함**.
3. **4→16으로 확장해도 추가 이득이 거의 없거나 오히려 손해.** 두 환경 모두 CPU async 기준
   n_envs=4에서 이미 처리량이 포화됨 (CartPole 2,000, Ant 1,333). GPU 조합은 16에서 오히려
   더 느려지는 경우도 있음(여러 프로세스가 물리 GPU 1개를 두고 경합하는 것으로 추정).
4. **결론**: 이 프레임워크(액터가 상태 1개씩 배치=1로 GPU에 왕복하는 구조) + 이 네트워크 크기에서는
   `use_cuda: False`, `n_envs=4, distributed_processing_type: "async"`가 두 환경 모두에서 실측상
   최적 지점. n_envs를 16까지 늘리는 건 이 벤치마크 기준으로는 이득이 없음.

### 다음에 비교할 것

- 더 큰 네트워크(이미지 관측 등)에서 GPU가 실제로 유리해지는 지점이 어디인지
- n_envs를 4~16 사이(8, 12 등)로 세분화해서 포화 지점이 정확히 어디인지
- AntBulletEnv를 원래 스텝 수(3,000,000)로 실제 학습 수렴까지 async n_envs=4로 완주 시 소요 시간

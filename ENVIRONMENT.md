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

- 2026-09-15: `ppo CartPole-v1` (n_envs=1, `Runner`) 벤치마크 진행 중 — 결과는 아래 대화/커밋 참고.

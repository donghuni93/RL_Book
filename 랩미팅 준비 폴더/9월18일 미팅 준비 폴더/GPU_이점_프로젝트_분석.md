# 어떤 프로젝트에서 GPU가 이점을 낼 수 있을까

CartPole 벤치마크에서 확인된 "batch=1 GPU 왕복 오버헤드" 문제를 바탕으로, 현재 레포에 있는
알고리즘·환경 조합들을 실제로 돌려보지 않고 이름/설정값만으로 분석한 기록.

- **날짜**: 2026-09-16
- **방법**: 코드 실행 없이 `config/agents/**/*.yaml`의 네트워크 크기, batch_size, 업데이트 구조만 확인
- **관련 기록**: `CartPole_벤치마크_노트.md`(GPU vs CPU 실측), `ENVIRONMENT.md`(전체 벤치마크 로그)

---

## 결론

지금 이 레포에 있는 알고리즘·환경 조합 중에는 **GPU가 확실히 이기는 조합이 없다.**
다만 이름/설정만 보고 순위를 매기면 상대적으로 가능성이 있는 순서는:

1. **DQN / DDQN** (off-policy, 배치 학습 구조)
2. **AntBulletEnv-v0** 계열 (물리 연산은 무겁지만 네트워크가 작음)
3. 나머지(REINFORCE, A2C, PPO on CartPole/LunarLander) — 거의 불가능

## 근거 — 확인한 config 값

| 알고리즘/환경 | 네트워크 크기 | 업데이트 구조 | GPU 가능성 |
|---|---|---|---|
| REINFORCE, A2C, PPO (CartPole-v1 / LunarLanderContinuous-v2) | 512 또는 64×64×64 | rollout마다 1스텝씩 batch=1 forward | 낮음 — 이미 실측으로 확인됨 (CPU가 최대 2.5배 빠름) |
| **DQN, DDQN** (CartPole-v1) | **256×256** (레포에서 가장 큰 네트워크) | `gradient_steps: 64`, `batch_size: 64`로 리플레이 버퍼에서 배치 학습을 몰아서 수행 | **상대적으로 가장 유리** |
| AntBulletEnv-v0 (PPO) | 64×64×64 | 동일하게 batch=1 rollout | 낮음 (물리 연산은 무겁지만 네트워크가 작음) |

**DQN/DDQN이 상대적으로 나은 이유**: 이 프레임워크의 근본 문제는 "매 스텝 상태 1개를 GPU에
왕복시키는 것"이다. DQN/DDQN은 off-policy라서 rollout(행동 선택)과 update(학습)가 분리되어 있다.
update 단계는 리플레이 버퍼에서 `batch_size=64`짜리 배치를 `gradient_steps=64`번 몰아서
학습하므로, 이 부분만 떼놓고 보면 진짜 "배치 연산"이라 GPU에 맞는 모양이다.

다만 지금 네트워크(256×256, 2층)와 배치(64) 규모로는 이것도 CPU가 이길 가능성이 높다 —
CartPole처럼 여전히 작은 편이라서다. rollout 쪽(행동 선택)은 DQN도 여전히 batch=1 방식이라
그대로 손해다.

## 새 프로젝트를 붙인다면 — GPU가 이기는 조건 3가지

관측(observation) / 네트워크 / 배치 중 **하나라도 왕복 오버헤드보다 계산량이 커야** GPU가 이긴다.

1. **이미지 기반 관측 환경** (예: Atari `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4` 류)
   지금 레포엔 없음 — `envs/opengym.py`가 CartPole/LunarLander/Acrobot/AntBulletEnv만 다루고,
   CNN 정책 코드 자체가 없다. 이미지 한 장의 conv 연산은 batch=1이어도 왕복 오버헤드를 가볍게
   넘어선다 — **가장 확실한 후보.**
2. **훨씬 큰 네트워크** (수천 유닛, 여러 층, LSTM/Transformer 기반 정책)
   지금 512/256/64 수준을 한 자릿수 이상 키우면 forward 연산량 자체가 커져서 GPU가 유리해진다.
3. **DQN/DDQN의 배치/gradient_steps를 크게 키우는 경우** (예: batch_size 512+, gradient_steps 수백)
   replay buffer 학습만 GPU로 보내고 rollout(행동 선택)은 CPU에 남기는 하이브리드 구성이면
   이득이 남는다.

## 한 줄 요약

> **"관측이 이미지거나, 네트워크/배치가 지금보다 훨씬 커야 GPU가 이긴다."**
> 이 프레임워크의 구조(`agents/actor.py`의 batch=1 forward)에서 나오는 일반 법칙이고,
> 지금 갖고 있는 어떤 조합도 그 문턱을 넘지 못한다.

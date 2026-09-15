# This file is part of RL_Book Project.
"""capture_gif.py: 강화학습 환경을 실행하며 프레임을 캡처해
Episode/Step/Return 정보를 텍스트로 새긴 GIF로 저장한다.

SSH 환경 등 화면(display)에 접근할 수 없는 곳에서도,
env.render(mode='rgb_array')로 프레임 배열만 얻어와 GIF로 저장하므로
GUI 창 없이 환경 동작을 눈으로 확인할 수 있다.
"""
import argparse
import os

import gym
from PIL import Image, ImageDraw, ImageFont


def get_font(size=18):
    """가능하면 트루타입 폰트를, 없으면 기본 폰트를 반환."""
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\malgun.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def capture_gif(env_name, out_path, n_steps=500, fps=25):
    """환경을 랜덤 행동으로 실행하며 프레임을 캡처해 GIF로 저장.

    Args:
        env_name: gym 환경 이름
        out_path: 저장할 gif 경로
        n_steps: 캡처할 최대 스텝 수
        fps: gif 초당 프레임 수
    """
    env = gym.make(env_name)
    font = get_font()

    frames = []
    episode = 1
    episode_return = 0.0

    env.action_space.seed(42)
    observation = env.reset()

    for step in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, env_info = env.step(action)
        episode_return += reward

        frame = env.render(mode='rgb_array')
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        text = f"Episode: {episode}  Step: {step}  Return: {episode_return:.1f}"
        # 가독성을 위해 텍스트에 검은 테두리(외곽선)를 그린다.
        x, y = 8, 6
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 0))
        frames.append(img)

        if done:
            episode += 1
            episode_return = 0.0
            observation = env.reset()

    env.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved {len(frames)} frames -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture gym environment as an annotated GIF')
    parser.add_argument('-e', '--env', type=str, default='CartPole-v1',
                         help='gym environment name')
    parser.add_argument('-s', '--steps', type=int, default=500,
                         help='number of steps to capture')
    parser.add_argument('-o', '--out', type=str,
                         default=os.path.join('results', 'captures', 'gym_capture.gif'),
                         help='output gif path (항상 results/captures 폴더가 기본값)')
    args = parser.parse_args()

    capture_gif(args.env, args.out, n_steps=args.steps)

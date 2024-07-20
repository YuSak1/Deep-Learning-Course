#!/usr/bin/env python3


import numpy as np
import gym
from gym import wrappers
from collections import deque

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=300, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.


    def bins(clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]


    def digitize_state(observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
            np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
            np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
            np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
        ]
        return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


    def get_action(next_state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if args.epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(q_table[next_state])
        else:
            next_action = np.random.choice([0, 1])
        return next_action


    # [3]1試行の各ステップの行動を保存しておくメモリクラス
    class Memory:
        def __init__(self, max_size=200):
            self.buffer = deque(maxlen=max_size)

        def add(self, experience):
            self.buffer.append(experience)

        def sample(self):
            return self.buffer.pop()  # 最後尾のメモリを取り出す

        def len(self):
            return len(self.buffer)


    # [4]Qテーブルを更新する(モンテカルロ法) ＊Qlearningと異なる＊ -------------------------------------
    def update_Qtable_montecarlo(q_table, memory):
        alpha = 0.5
        total_reward_t = 0

        while (memory.len() > 0):
            (state, action, reward) = memory.sample()
            total_reward_t = args.gamma * total_reward_t       # 時間割引率をかける
            # Q関数を更新
            q_table[state, action] = q_table[state, action] + alpha*(reward+total_reward_t-q_table[state, action])
            total_reward_t = total_reward_t + reward    # ステップtより先でもらえた報酬の合計を更新

        return q_table


    env = gym.make('CartPole-v0')
    max_number_of_steps = 200  #1試行のstep数
    num_episodes = args.episodes
    goal_average_reward = 195  #この報酬を超えると学習終了（中心への制御なし）
    # 状態を6分割^（4変数）にデジタル変換してQ関数（表）を作成
    num_dizitized = 6  #分割数
    memory_size = max_number_of_steps            # バッファーメモリの大きさ
    memory = Memory(max_size=memory_size)
    q_table = np.random.uniform(low=-1, high=1, size=(num_dizitized**4, env.action_space.n))
    total_reward_vec = np.zeros(100)  #各試行の報酬を格納
    final_x = np.zeros((num_episodes, 1))  #学習後、各試行のt=200でのｘの位置を格納
    islearned = 0  #学習が終わったフラグ
    isrender = 0  #描画フラグ


    for episode in range(num_episodes):
        observation = env.reset()
        state = digitize_state(observation)
        action = np.argmax(q_table[state])
        episode_reward = 0

        for t in range(max_number_of_steps):
            if islearned == 1: 
                env.render()
                print (observation[0])

            observation, reward, done, info = env.step(action)

            # Reward
            if done:
                if t < 195:
                    reward = -200
                else:
                    reward = 1
            else:
                reward = 1

            memory.add((state, action, reward))

            next_state = digitize_state(observation)
            next_action = get_action(next_state, episode)
            action = next_action
            state = next_state
            episode_reward += reward

            if done:
                # これまでの行動の記憶と、最終的な結果からQテーブルを更新していく
                q_table = update_Qtable_montecarlo(q_table, memory)

                print('%d Episode, Mean: %f' %
                      (episode, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:],
                                              episode_reward))  #報酬を記録
                if islearned == 1:  #学習終わってたら最終のx座標を格納
                    final_x[episode, 0] = observation[0]
                break


    np.savetxt('final_x.csv', final_x, delimiter=",")
    isrender = 1



    while training:
        # Perform a training episode
        state, done = env.reset(), False # start_evaluate=True
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            next_state, reward, done, _ = env.step(action)

    # Perform last 100 evaluation episodes
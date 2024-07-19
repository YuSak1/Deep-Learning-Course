import numpy as np
import gym
import mountain_car_evaluator

# alpha = np.exp(np.log(args.alpha) * prog + np.log(args.alpha_final) * (1-prog)) 
 
# def get_status(_observation):
#     posi_low = -1.2
#     posi_high = 0.6
#     posi_d = (posi_high - posi_low) / 40
#     velo_low = -0.07
#     velo_high = 0.07
#     velo_d = (velo_high - velo_low) / 40
    
#     # 0〜39の離散値に変換する
#     position = int((_observation[0] - posi_low) / posi_d)
#     velocity = int((_observation[1] - velo_low) / velo_d)
#     return position, velocity

def get_status(_observation):
    env_low = env.observation_space.low # 位置と速度の最小値
    env_high = env.observation_space.high #　位置と速度の最大値
    env_dx = (env_high - env_low) / 40 # 40等分
    # 0〜39の離散値に変換する
    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity


q_table = np.zeros((40, 40, 3))

def update_q_table(_q_table, _action,  _observation, _next_observation, _reward, _episode):

    alpha = 0.1 # 学習率
    gamma = 0.99 # 時間割引き率

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_position, next_velocity = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocity])

    # 行動前の状態の行動価値 Q(s,a)
    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]

    # 行動価値関数の更新
    _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table


def get_action(_env, _q_table, _observation, _episode):
    epsilon = 0.001
    epsilon_final = 0.000001
    # eps = np.exp(np.log(epsilon) * (10000/_episode) + np.log(epsilon_final) * (1-(10000/_episode)))

    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(observation)
        _action = np.argmax(_q_table[position][velocity])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action


if __name__ == '__main__':

    # Create the environment
    # env = mountain_car_evaluator.environment()
    env = gym.make('MountainCar-v0')
    # env.render()

    # Initial q_table
    q_table = np.zeros((40, 40, 3))

    observation = env.reset()

    reward_target = -130
    rewards = []

    # 10000エピソードで学習する
    training = True
    while training:
        for episode in range(10000):

            total_reward = 0
            observation = env.reset()

            for _ in range(200):
                action = get_action(env, q_table, observation, episode)

                # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
                next_observation, reward, done, _ = env.step(action)

                q_table = update_q_table(q_table, action, observation, next_observation, reward, episode)
                total_reward += reward

                observation = next_observation

                if done:
                    # doneがTrueになったら１エピソード終了
                    if episode%100 == 0:
                        print('episode: {}, total_reward: {}'.format(episode, total_reward))
                    rewards.append(total_reward)
                    break
                
                # Stop learning if already ready to run
                if total_reward > reward_target:
                    training = False


    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = get_action(env, q_table, observation, episode)
            next_state, reward, done, _ = env.step(action)
            state = next_state
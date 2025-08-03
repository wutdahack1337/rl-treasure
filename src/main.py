from tqdm import tqdm

import utils
from config  import TrainingConfig
from treasure_env   import TreasureEnv
from treasure_agent import TreasureAgent

def main():
    config = TrainingConfig()

    env   = TreasureEnv(config.size, config.seed)
    agent = TreasureAgent(env, config.learning_rate, config.epsilon, config.discount_factor, config.seed)

    step_cnt_queue = train_agent(env, agent, config.train_episodes)

    utils.check_q_values(env, agent)
    utils.plot_training_results(step_cnt_queue, config.size, config.train_episodes)

    test_agent(env, agent, config.test_episodes)

def train_agent(env, agent, n_episodes):
    step_cnt_queue = []
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, info = env.reset()
        step_cnt = 0
        terminated = False

        while not terminated:
            action = agent.get_action(obs, verbose=0)
            next_obs, reward, terminated, info = env.step(action)
            step_cnt += 1

            agent.learn(obs, action, reward, terminated, next_obs)
            obs = next_obs

        step_cnt_queue.append(step_cnt)

    return step_cnt_queue
    
def test_agent(env, agent, test_episodes):
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.0

    for episode in range(test_episodes):
        obs, info = env.reset()
        print(f"=== Test episode {episode + 1} ===")
        utils.check_response(env, obs, 0)

        terminated = False
        while not terminated:
            action = agent.get_action(obs, verbose=0)
            obs, reward, terminated, info = env.step(action)
            utils.check_response(env, obs, reward)

    agent.epsilon = original_epsilon

if __name__ == "__main__":
    main()
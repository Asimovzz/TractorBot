from multiprocessing import Process
import numpy as np
import torch
import os
import glob
import random
import time

from tractorbot.rl.replay_buffer import ReplayBuffer
from tractorbot.rl.model_pool import ModelPoolClient
from tractorbot.envs.env import TractorEnv
from tractorbot.models.model import CNNModel
from tractorbot.envs.wrapper import cardWrapper

try:
    from tractorbot.agents.better_bot import BetterBot
except ImportError:
    BetterBot = None
    print("Warning: tractorbot.agents.better_bot not found.")

try:
    from tractorbot.agents.heu_train_bot import HeuBot
except ImportError:
    HeuBot = None
    print("Warning: tractorbot.agents.heu_bot not found.")

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
        
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        main_model = CNNModel()      
        opponent_model = CNNModel()
        
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        main_model.load_state_dict(state_dict)
        
        opponent_library = {} 
        bots_library = {}
        
        legend_paths = {
            # 
        }
        
        for name, path in legend_paths.items():
            if os.path.exists(path):
                try:
                    w = torch.load(path, map_location='cpu', weights_only=True)
                    opponent_library[name] = w
                    print(f"[{self.name}] Loaded Legend Model: {name}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")
        
        if BetterBot:
            bots_library['better_bot'] = BetterBot()
        if HeuBot:
            bots_library['heu_bot'] = HeuBot(level='2')
            print(f"[{self.name}] HeuBot (Expert) Ready.")
            
        env = TractorEnv()
        self.wrapper = cardWrapper()
        
        reward_buffer = []
        exp_cfg = self.config.get('experiment', {})
        SAVE_INTERVAL = exp_cfg.get('save_interval', 20)
        save_dir = os.path.join(exp_cfg.get('save_dir', 'logs'), exp_cfg.get('name', 'default_exp'))
        os.makedirs(save_dir, exist_ok=True)
        full_save_path = os.path.join(save_dir, f"reward_{self.name}.txt")

        episode = 0
        
        while True:
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                if state_dict is not None:
                    main_model.load_state_dict(state_dict)
                    version = latest
        
            rand_val = random.random()
            opponent_type = "unknown"
            current_bot = None
            use_neural_opponent = False
            
            opp_cfg = self.config.get('opponents', {})
            prob_heu = opp_cfg.get('heu_bot_prob', 0.05)
            prob_neural = opp_cfg.get('neural_bot_prob', 0.60) + prob_heu
            prob_better = opp_cfg.get('better_bot_prob', 0.25) + prob_neural

            if rand_val < prob_heu and 'heu_bot' in bots_library:
                opponent_type = "heu_bot"
                current_bot = bots_library['heu_bot']
                
            elif rand_val < prob_neural and len(opponent_library) > 0:
                opp_name = '1935'
                opponent_model.load_state_dict(opponent_library['1935'])
                opponent_type = "model_1935"
                use_neural_opponent = True
                
            elif rand_val < prob_better:
                opponent_type = "better_bot"
                current_bot = bots_library['better_bot']

            else:
                opponent_type = "self_play"
                opponent_model.load_state_dict(main_model.state_dict())
                use_neural_opponent = True

            obs, action_options = env.reset(major='r')
            
            episode_data = {
                'player_0': {'state': {'observation': [], 'global_feature': [], 'action_mask': []}, 'action': [], 'reward': [], 'value': [], 'log_prob': []},
                'player_2': {'state': {'observation': [], 'global_feature': [], 'action_mask': []}, 'action': [], 'reward': [], 'value': [], 'log_prob': []}
            }
            
            done = False
            while not done:
                player = obs['id']
                agent_name = env.agent_names[player]
                
                if player in [0, 2]:
                    wrapped_obs = self.wrapper.obsWrap(obs, action_options)
                    obs_mat = wrapped_obs['observation']
                    global_feat = wrapped_obs['global_feature']
                    action_mask = wrapped_obs['action_mask']
                    
                    episode_data[agent_name]['state']['observation'].append(obs_mat)
                    episode_data[agent_name]['state']['global_feature'].append(global_feat)
                    episode_data[agent_name]['state']['action_mask'].append(action_mask)
                    
                    state_tensor = {
                        'observation': torch.tensor(obs_mat, dtype=torch.float).unsqueeze(0),
                        'global_feature': torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),
                        'action_mask': torch.tensor(action_mask, dtype=torch.float).unsqueeze(0)
                    }
                    
                    main_model.train(False)
                    with torch.no_grad():
                        logits, value = main_model(state_tensor)
                        action_dist = torch.distributions.Categorical(logits=logits)
                        action_tensor = action_dist.sample()
                        log_prob = action_dist.log_prob(action_tensor).item()
                        action = action_tensor.item()
                        value = value.item()
                    
                    episode_data[agent_name]['action'].append(action)
                    episode_data[agent_name]['log_prob'].append(log_prob)
                    episode_data[agent_name]['value'].append(value)
                
                else:
                    action = 0
                    
                    if use_neural_opponent:
                        wrapped_obs = self.wrapper.obsWrap(obs, action_options)
                        opp_state = {
                            'observation': torch.tensor(wrapped_obs['observation'], dtype=torch.float).unsqueeze(0),
                            'global_feature': torch.tensor(wrapped_obs['global_feature'], dtype=torch.float).unsqueeze(0),
                            'action_mask': torch.tensor(wrapped_obs['action_mask'], dtype=torch.float).unsqueeze(0)
                        }
                        opponent_model.train(False)
                        with torch.no_grad():
                            logits, _ = opponent_model(opp_state)
                            action_dist = torch.distributions.Categorical(logits=logits)
                            action = action_dist.sample().item()
                    else:
                        action = current_bot.step(obs, action_options)
                        
                action_cards = action_options[action]
                response = env.action_intpt(action_cards, player)
                next_obs, action_options, rewards, done = env.step(response)
                
                if rewards:
                    for target_agent in ['player_0', 'player_2']:
                        if target_agent in rewards:
                            episode_data[target_agent]['reward'].append(rewards[target_agent])
                            
                obs = next_obs

            for agent_name in ['player_0', 'player_2']:
                agent_data = episode_data[agent_name]
                
                min_len = min(len(agent_data['action']), len(agent_data['reward']))
                if min_len == 0: continue
                
                obs_np = np.stack(agent_data['state']['observation'][:min_len])
                global_np = np.stack(agent_data['state']['global_feature'][:min_len])
                mask_np = np.stack(agent_data['state']['action_mask'][:min_len])
                actions_np = np.array(agent_data['action'][:min_len], dtype=np.int64)
                reward_scale = self.config.get('env', {}).get('rewards', {}).get('scale_factor', 10.0)
                rewards_np = np.array(agent_data['reward'][:min_len], dtype=np.float32) / reward_scale
                values_np = np.array(agent_data['value'][:min_len], dtype=np.float32)
                log_probs_np = np.array(agent_data['log_prob'][:min_len], dtype=np.float32)
                
                next_values = np.concatenate([values_np[1:], [0]])
                td_target = rewards_np + next_values * self.config['gamma']
                td_delta = td_target - values_np
                
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv)
                advs.reverse()
                advantages = np.array(advs, dtype=np.float32)
                
                self.replay_buffer.push({
                    'state': {'observation': obs_np, 'global_feature': global_np, 'action_mask': mask_np},
                    'action': actions_np,
                    'adv': advantages,
                    'target': td_target,
                    'log_prob': log_probs_np
                })

            p0_r = np.sum(episode_data['player_0']['reward']) if episode_data['player_0']['reward'] else 0
            p2_r = np.sum(episode_data['player_2']['reward']) if episode_data['player_2']['reward'] else 0
            my_score = (p0_r + p2_r) / 2.0
            
            reward_buffer.append(my_score)
            
            if (episode + 1) % SAVE_INTERVAL == 0:
                avg_score = np.mean(reward_buffer)
                print(f"[{self.name}] Ep {episode}: vs {opponent_type}, AvgScore {avg_score:.2f}")
                try:
                    with open(full_save_path, 'a') as f:
                        f.write(f"{episode},{opponent_type},{avg_score:.4f}\n")
                    reward_buffer = []
                except: pass
            
            episode += 1
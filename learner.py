from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import os

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
    
    def run(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        log_dir = os.path.join(base_dir, 'logs/exp_league_1')
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")
            except OSError as e:
                print(f"Error creating log directory: {e}")
        
        loss_log_path = os.path.join(log_dir, 'loss_log.csv')
        
        try:
            with open(loss_log_path, 'w') as f:
                f.write("Iteration,PolicyLoss,ValueLoss,EntropyLoss,TotalLoss\n")
            print(f"Logging loss to: {loss_log_path}")
        except Exception as e:
            print(f"Error opening log file: {e}")

        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        device = torch.device(self.config['device'])
        model = CNNModel()
        
        pretrained_path = os.path.join(base_dir, 'history_model', 'model_phase2.pt')
        if os.path.exists(pretrained_path):
            print(f"Loading Model: {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        model_pool.push(model.state_dict()) 
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'], eps=1e-5)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.9)
        
        print(f"Learner waiting for {self.config['min_sample']} samples...")
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(1)
        
        print("Learner starting training loop...")
        cur_time = time.time()
        iterations = 0
        initial_entropy_coeff = self.config['entropy_coeff']
        
        torch.backends.cudnn.benchmark = False
        
        while True:
            
            current_size = self.replay_buffer.size()
            while current_size < self.config['batch_size']:
                print(f"Waiting for samples... {current_size}/{self.config['batch_size']}")
                time.sleep(2)
                current_size = self.replay_buffer.size()
            
            batch = self.replay_buffer.sample(self.config['batch_size'])
            
            obs_all = batch['state']['observation']
            global_all = batch['state']['global_feature']
            mask_all = batch['state']['action_mask']
            actions_all = batch['action']
            advs_all = batch['adv']
            targets_all = batch['target']
            
            old_log_probs_all = batch['log_prob'] 
            
            # 优势归一化 
            advs_all = (advs_all - advs_all.mean()) / (advs_all.std() + 1e-8)
            
            print('Iteration %d, Buffer: %d' % (iterations, self.replay_buffer.stats['sample_in']))
            
            epoch_loss_stats = {'policy': [], 'value': [], 'entropy': [], 'total': []}

            for _ in range(self.config['epochs']):
                total_samples = obs_all.shape[0]
                indices = np.arange(total_samples)
                np.random.shuffle(indices)
                
                mini_batch_size = self.config.get('mini_batch_size', 64)
                
                for start_idx in range(0, total_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, total_samples)
                    mb_indices = indices[start_idx:end_idx]
                    
                    obs = torch.tensor(obs_all[mb_indices], dtype=torch.float).to(device)
                    global_feat = torch.tensor(global_all[mb_indices], dtype=torch.float).to(device)
                    mask = torch.tensor(mask_all[mb_indices], dtype=torch.float).to(device)
                    actions = torch.tensor(actions_all[mb_indices], dtype=torch.int64).unsqueeze(-1).to(device)
                    advs = torch.tensor(advs_all[mb_indices], dtype=torch.float).to(device)
                    targets = torch.tensor(targets_all[mb_indices], dtype=torch.float).to(device)
                    old_log_probs = torch.tensor(old_log_probs_all[mb_indices], dtype=torch.float).to(device)
                    
                    states = {'observation': obs, 'global_feature': global_feat, 'action_mask': mask}
                    
                    model.train(True)
                    
                    logits, values = model(states)
                    
                    action_dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = action_dist.log_prob(actions.squeeze(-1))
                    entropy = action_dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    
                    value_loss = F.mse_loss(values.squeeze(-1), targets)
                    
                    decay_rate = 0.9995
                    current_entropy_coeff = max(0.001, initial_entropy_coeff * (decay_rate ** iterations))
                    
                    loss = policy_loss + self.config['value_coeff'] * value_loss - current_entropy_coeff * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    epoch_loss_stats['policy'].append(policy_loss.item())
                    epoch_loss_stats['value'].append(value_loss.item())
                    epoch_loss_stats['entropy'].append(-entropy.item())
                    epoch_loss_stats['total'].append(loss.item())
                    
                    del obs, mask, actions, advs, targets, old_log_probs, logits, values, loss

            avg_policy = np.mean(epoch_loss_stats['policy'])
            avg_value = np.mean(epoch_loss_stats['value'])
            avg_entropy = np.mean(epoch_loss_stats['entropy'])
            avg_total = np.mean(epoch_loss_stats['total'])
            
            with open(loss_log_path, 'a') as f:
                f.write(f"{iterations},{avg_policy:.4f},{avg_value:.4f},{avg_entropy:.4f},{avg_total:.4f}\n")

            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            model_pool.push(cpu_state_dict) 
            
            self.replay_buffer.clear()
            
            scheduler.step()
            
            if iterations % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Current LR: {current_lr:.2e}")
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(model.state_dict(), path)
                cur_time = t
            
            iterations += 1
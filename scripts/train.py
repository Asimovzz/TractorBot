import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tractorbot.rl.replay_buffer import ReplayBuffer
from tractorbot.rl.actor import Actor
from tractorbot.rl.learner import Learner
import torch

if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    config = {
        'replay_buffer_size': 100000,
        'replay_buffer_episode': 2000,
        'model_pool_size': 5,
        'model_pool_name': 'model-pool',
        'num_actors': 275,
        'episodes_per_actor': 1000,
        'gamma': 0.99,  
        'lambda': 0.95,
        'min_sample': 4096,
        'batch_size': 4096,  
        'mini_batch_size': 1024,
        'epochs': 10,
        'clip': 0.2, 
        'lr': 3e-4,  
        'value_coeff': 0.5,    
        'entropy_coeff': 0.04,   
        'device': 'cuda',
        'ckpt_save_interval': 500,
        'ckpt_save_path': 'checkpoints/'
    }
    
    if not os.path.exists(config['ckpt_save_path']):
        os.makedirs(config['ckpt_save_path'])
        print(f"Created directory: {config['ckpt_save_path']}")
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
        
    learner = Learner(config, replay_buffer)
    
    print("Starting Learner...")
    learner.start()
    
    import time
    time.sleep(3)
    
    print(f"Starting {config['num_actors']} Actors...")
    for actor in actors: 
        actor.start()
        time.sleep(0.1)
    
    for actor in actors: actor.join()
    learner.terminate()
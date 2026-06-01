import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import time

import torch

from tractorbot.config import build_training_config, load_config
from tractorbot.rl.replay_buffer import ReplayBuffer
from tractorbot.rl.actor import Actor
from tractorbot.rl.learner import Learner


def parse_args():
    parser = argparse.ArgumentParser(description="Train TractorBot with actor-learner self-play.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. Defaults to configs/default.yaml or TRACTORBOT_CONFIG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_training_config(load_config(args.config))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    time.sleep(config.get('actor_start_delay_seconds', 3.0))
    
    print(f"Starting {config['num_actors']} Actors...")
    for actor in actors: 
        actor.start()
        time.sleep(config.get('actor_start_interval_seconds', 0.1))
    
    for actor in actors: actor.join()
    learner.terminate()


if __name__ == '__main__':
    main()

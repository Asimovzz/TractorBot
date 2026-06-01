from multiprocessing import Queue
from collections import deque
import numpy as np
import random

class ReplayBuffer:

    def __init__(self, capacity, episode):
        self.queue = Queue(episode)
        self.capacity = capacity
        self.buffer = None
    
    def push(self, samples):
        self.queue.put(samples)
    
    def _flush(self):
        if self.buffer is None:
            self.buffer = deque(maxlen = self.capacity)
            self.stats = {'sample_in': 0, 'sample_out': 0, 'episode_in': 0}
        while not self.queue.empty():
            episode_data = self.queue.get()
            unpacked_data = self._unpack(episode_data)
            self.buffer.extend(unpacked_data)
            self.stats['sample_in'] += len(unpacked_data)
            self.stats['episode_in'] += 1
    
    def sample(self, batch_size):
        self._flush()
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats['sample_out'] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch
    
    def size(self):
        self._flush()
        return len(self.buffer)
    
    def clear(self):
        self._flush()
        self.buffer.clear()
    
    def _unpack(self, data):
        # Convert nested episode arrays into per-step samples.
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res: res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)
            
    def _pack(self, data):
        # Rebuild a nested batch dictionary from sampled steps.
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)

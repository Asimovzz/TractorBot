import random

class RuleBot:
    def __init__(self):
        pass

    def step(self, obs, action_options):
        if len(action_options) == 1:
            return 0
        
        best_action_idx = 0
        
        max_score = -1
        
        for i, opt in enumerate(action_options):
            score = 0
            score += len(opt) * 10 
            
            for card in opt:
                if 'A' in card: score += 5
                if 'K' in card: score += 3
                if 'jo' in card or 'Jo' in card: score += 6
            
            if score > max_score:
                max_score = score
                best_action_idx = i
                
        return best_action_idx
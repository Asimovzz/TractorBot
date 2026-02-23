import random

class BetterBot:
    def __init__(self):
        self.point_cards = ['5', '0', 'K']

    def _get_card_score(self, card_name):
        score = 0
        if card_name[1] in self.point_cards: score += 10
        if 'A' in card_name: score += 8
        if 'jo' in card_name or 'Jo' in card_name: score += 15
        return score

    def step(self, obs, action_options):
        if len(action_options) == 1:
            return 0
            
        history = obs['history']
        am_i_leader = (len(history) == 0)
        
        if am_i_leader:
            best_idx = 0
            max_power = -1
            
            for i, opt in enumerate(action_options):
                power = 0
                power += len(opt) * 100
                
                is_boss = all(x[1] in ['A', 'K', 'o'] for x in opt)
                if is_boss: power += 50
                
                has_point = any(x[1] in self.point_cards for x in opt)
                if has_point and not is_boss: power -= 30
                
                if power > max_power:
                    max_power = power
                    best_idx = i
            return best_idx

        # 跟牌
        else:
            current_winner_idx = 0
            
            my_seat = obs['id']
            
            best_idx = 0
            best_val = -9999
            
            for i, opt in enumerate(action_options):
                val = 0
                cards = opt
                
                points = sum([1 for c in cards if c[1] in self.point_cards])
                
                score_lost = sum([self._get_card_score(c) for c in cards])
                val -= score_lost
                
                if val > best_val:
                    best_val = val
                    best_idx = i
                    
            return best_idx
import random

'''
该HeuBot用作本地训练
'''
class HeuBot:
    def __init__(self, level='2'):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['s','h','c','d']
        self.pointorder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
        self.base_level = level
        
        self.GlobalLevel = '2'
        self.GlobalMajorSuit = 'n'
        self.GlobalMajorList = [] 
        self.GlobalPointOrder = []

    def _update_global_consts(self, major_list):
        counts = {}
        for c in major_list:
            if c in ['jo', 'Jo']: continue
            rank = c[1]
            counts[rank] = counts.get(rank, 0) + 1
        
        if counts:
            self.GlobalLevel = max(counts, key=counts.get)
        else:
            self.GlobalLevel = self.base_level

        self.GlobalMajorSuit = 'n'
        for c in major_list:
            if c in ['jo', 'Jo']: continue
            if c[1] != self.GlobalLevel:
                self.GlobalMajorSuit = c[0]
                break
        
        self.GlobalPointOrder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
        if self.GlobalLevel in self.GlobalPointOrder:
            self.GlobalPointOrder.remove(self.GlobalLevel)
        
        self.GlobalMajorList = major_list

    def _get_suit(self, card_name):
        if card_name in self.GlobalMajorList: return 'master'
        return card_name[0]

    def _get_rank_weight(self, card_name):
        if card_name == 'Jo': return 200
        if card_name == 'jo': return 190
        if card_name[1] == self.GlobalLevel:
            if card_name[0] == self.GlobalMajorSuit: return 180 
            return 170 
        if self.GlobalMajorSuit != 'n' and card_name[0] == self.GlobalMajorSuit:
            try: return 100 + self.GlobalPointOrder.index(card_name[1])
            except: return 100
        try: return self.GlobalPointOrder.index(card_name[1])
        except: return 0

    def _get_point_value(self, card_name):
        r = card_name[1]
        if r == '5': return 5
        if r == '0' or r == 'K': return 10
        return 0

    def step(self, obs, action_options):
        if len(action_options) == 1:
            return 0

        self._update_global_consts(obs['major'])
        
        my_id = obs['id']
        history_curr = obs['history']
        played_history = obs['played']
        
        void_map = [{s: False for s in self.suitset + ['master']} for _ in range(4)]
        
        if len(history_curr) > 0:
            leader_move = history_curr[0]
            leader_suit = self._get_suit(leader_move[0])
            leader_id = (my_id - len(history_curr)) % 4
            
            for i in range(1, len(history_curr)):
                player_curr = (leader_id + i) % 4
                move = history_curr[i]
                actual_suit = self._get_suit(move[0])
                if actual_suit != leader_suit:
                    void_map[player_curr][leader_suit] = True

        if len(history_curr) == 0:
            best_idx = 0
            best_score = -99999
            
            for i, move in enumerate(action_options):
                score = 0
                card = move[0]
                suit = self._get_suit(card)
                length = len(move)
                weight = self._get_rank_weight(card)
                
                if suit == 'master':
                    score += 500
                    if weight > 150: score += 50
                    if length > 1: score += 200 * length
                else:
                    if weight > 10:
                        score += 100
                    if length > 1: 
                        score += 150 * length
                    
                    points = sum(self._get_point_value(c) for c in move)
                    if points > 0 and weight < 10:
                        score -= 200
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            return best_idx

        else:
            leader_move = history_curr[0]
            leader_suit = self._get_suit(leader_move[0])
            
            winning_weight = self._get_rank_weight(leader_move[0])
            is_ruffed = (leader_suit == 'master')
            winner_rel_idx = 0
            
            for k in range(1, len(history_curr)):
                c = history_curr[k][0]
                s = self._get_suit(c)
                w = self._get_rank_weight(c)
                if s == leader_suit and not is_ruffed:
                    if w > winning_weight:
                        winning_weight = w
                        winner_rel_idx = k
                elif s == 'master' and leader_suit != 'master':
                    if not is_ruffed:
                        winning_weight = w
                        is_ruffed = True
                        winner_rel_idx = k
                    elif w > winning_weight:
                        winning_weight = w
                        winner_rel_idx = k
            
            leader_id = (my_id - len(history_curr)) % 4
            current_winner_id = (leader_id + winner_rel_idx) % 4
            partner_id = (my_id + 2) % 4
            we_are_winning = (current_winner_id == partner_id) or (current_winner_id == my_id)
            
            best_idx = 0
            best_score = -99999
            
            for i, move in enumerate(action_options):
                score = 0
                card = move[0]
                suit = self._get_suit(card)
                weight = self._get_rank_weight(card)
                points = sum(self._get_point_value(c) for c in move)
                
                can_beat = False
                if suit == leader_suit and not is_ruffed:
                    if weight > winning_weight: can_beat = True
                elif suit == 'master' and leader_suit != 'master':
                    if not is_ruffed: can_beat = True
                    elif weight > winning_weight: can_beat = True
                
                if we_are_winning:
                    score += points * 50
                    score -= weight
                else:
                    if can_beat:
                        score += 1000 
                        score -= weight
                    else:
                        if points > 0: score -= 1000
                        score -= weight
                        
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            return best_idx
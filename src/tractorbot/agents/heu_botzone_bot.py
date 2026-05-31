import sys
from collections import Counter

cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
suitset = ['s','h','c','d']
pointorder_tpl = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']

def Num2Poker(num): 
    if isinstance(num, str): return num 
    NumInDeck = num % 54
    if NumInDeck == 52: return "jo"
    if NumInDeck == 53: return "Jo"
    return suitset[NumInDeck % 4] + cardscale[NumInDeck // 4]

def map_action_to_hold_ids(action_names, current_hold_ids):
    result_ids = []
    temp_hold = current_hold_ids[:] 
    for name in action_names:
        found = False
        for pid in temp_hold:
            if Num2Poker(pid) == name:
                result_ids.append(pid)
                temp_hold.remove(pid)
                found = True
                break
        if not found:
            if temp_hold:
                fallback = temp_hold.pop(0)
                result_ids.append(fallback)
    return result_ids

class GameState:
    def __init__(self, self_id, full_requests, current_hold, major_list):
        self.my_id = self_id
        self.major_list = major_list
        self.unknown_cards = []
        
        for d in range(2):
            for s in suitset:
                for r in cardscale: self.unknown_cards.append(s+r)
            self.unknown_cards.append("jo"); self.unknown_cards.append("Jo")
        
        my_hand_names = [Num2Poker(x) for x in current_hold]
        for c in my_hand_names:
            if c in self.unknown_cards: self.unknown_cards.remove(c)
            
        self.void_map = [{s: False for s in suitset + ['master']} for _ in range(4)]
        self.replay_history(full_requests)

    def get_suit(self, card_name):
        if card_name in self.major_list: return 'master'
        return card_name[0]

    def get_rank_weight(self, card_name, global_level, global_major_suit, global_point_order):
        if card_name == 'Jo': return 200
        if card_name == 'jo': return 190
        if card_name[1] == global_level:
            if card_name[0] == global_major_suit: return 180 
            return 170 
        if global_major_suit != 'n' and card_name[0] == global_major_suit:
            try: return 100 + global_point_order.index(card_name[1])
            except: return 100
        try: return global_point_order.index(card_name[1])
        except: return 0

    def replay_history(self, requests):
        for req in requests:
            if req["stage"] == "play":
                history = req["history"]
                prev_round = history[0]
                leader_id = history[2]
                if prev_round:
                    self.analyze_round(prev_round, leader_id)
                
                curr_round = history[1]
                if curr_round:
                    for move in curr_round:
                        for cid in move:
                            cname = Num2Poker(cid)
                            if cname in self.unknown_cards: self.unknown_cards.remove(cname)

    def analyze_round(self, round_cards, leader_id):
        if not round_cards or len(round_cards) != 4: return
        leader_move = [Num2Poker(x) for x in round_cards[0]]
        if not leader_move: return
        leader_suit = self.get_suit(leader_move[0])
        
        for i in range(4):
            player_id = (leader_id + i) % 4
            move_ids = round_cards[i]
            move_names = [Num2Poker(x) for x in move_ids]
            
            for name in move_names:
                if name in self.unknown_cards: self.unknown_cards.remove(name)
            
            if i > 0: 
                actual_suit = self.get_suit(move_names[0])
                if actual_suit != leader_suit:
                    self.void_map[player_id][leader_suit] = True

    def is_safe_master(self, card_name, global_level, global_major_suit, global_point_order):
        suit = self.get_suit(card_name)
        my_weight = self.get_rank_weight(card_name, global_level, global_major_suit, global_point_order)
        
        for c in self.unknown_cards:
            if self.get_suit(c) == suit:
                if self.get_rank_weight(c, global_level, global_major_suit, global_point_order) > my_weight: 
                    return False, "NotBiggest"
        
        if suit != 'master':
            enemies = [(self.my_id + 1) % 4, (self.my_id + 3) % 4]
            for enemy in enemies:
                if self.void_map[enemy][suit]:
                    has_trump_left = any(self.get_suit(c) == 'master' for c in self.unknown_cards)
                    if has_trump_left: return False, "RiskRuff"
        return True, "Safe"


class HeuBot:
    def __init__(self, level='2'):
        self.base_level = level 

    def get_point_value(self, card_name):
        r = card_name[1]
        if r == '5': return 5
        if r == '0' or r == 'K': return 10
        return 0

    def step(self, full_requests, hold, self_id, history_curr, action_options, global_level, global_major):
        GlobalLevel = global_level
        GlobalMajorSuit = global_major
        GlobalPointOrder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
        if GlobalLevel in GlobalPointOrder: GlobalPointOrder.remove(GlobalLevel)
        
        GlobalMajorList = ['jo', 'Jo']
        if GlobalMajorSuit != 'n':
            GlobalMajorList = [GlobalMajorSuit + GlobalLevel] + GlobalMajorList 
            others = [s + GlobalLevel for s in suitset if s != GlobalMajorSuit]
            GlobalMajorList = others + GlobalMajorList 
            trump_seq = [GlobalMajorSuit + p for p in GlobalPointOrder]
            GlobalMajorList = trump_seq + GlobalMajorList
        else:
            all_levels = [s + GlobalLevel for s in suitset]
            GlobalMajorList = all_levels + GlobalMajorList

        state = GameState(self_id, full_requests, hold, GlobalMajorList)
        
        hold_names = [Num2Poker(x) for x in hold]
        
        leader_suit = None
        winning_weight = -1
        is_ruffed = False
        
        if history_curr:
            leader_card = Num2Poker(history_curr[0][0])
            leader_suit = state.get_suit(leader_card)
            winning_weight = state.get_rank_weight(leader_card, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
            for i in range(1, len(history_curr)):
                c = Num2Poker(history_curr[i][0])
                s = state.get_suit(c)
                w = state.get_rank_weight(c, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
                if s == leader_suit and not is_ruffed:
                    if w > winning_weight: winning_weight = w
                elif s == 'master' and leader_suit != 'master':
                    if not is_ruffed: winning_weight = w; is_ruffed = True
                    else: 
                        if w > winning_weight: winning_weight = w

        if not history_curr:
            best_score = -99999
            best_idx = 0
            
            for idx, move in enumerate(action_options):
                score = 0
                card = move[0]
                suit = state.get_suit(card)
                length = len(move)
                is_master, status = state.is_safe_master(card, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
                
                if is_master:
                    score += 500
                    if length > 1: score += 100 * length
                    points = sum(self.get_point_value(c) for c in move)
                    score += points * 5
                else:
                    if status == "RiskRuff": score -= 500 
                    elif status == "NotBiggest":
                        w = state.get_rank_weight(card, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
                        if w < 20: 
                            count_in_hand = sum(1 for c in hold_names if state.get_suit(c) == suit)
                            if count_in_hand == length: score += 50 
                            else: score -= 50 
                        elif w > 100: score -= 100
                
                if suit == 'master':
                    if state.get_rank_weight(card, GlobalLevel, GlobalMajorSuit, GlobalPointOrder) > 150: score += 40
                    else: score -= 20 
                    
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        else:
            leader_id = (state.my_id - len(history_curr)) % 4
            winner_rel_idx = 0
            temp_win_w = state.get_rank_weight(Num2Poker(history_curr[0][0]), GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
            temp_ruff = False
            
            for i in range(1, len(history_curr)):
                c = Num2Poker(history_curr[i][0])
                s = state.get_suit(c)
                w = state.get_rank_weight(c, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
                if s == leader_suit and not temp_ruff:
                    if w > temp_win_w: temp_win_w = w; winner_rel_idx = i
                elif s == 'master' and leader_suit != 'master':
                    if not temp_ruff: temp_win_w = w; winner_rel_idx = i; temp_ruff = True
                    elif w > temp_win_w: temp_win_w = w; winner_rel_idx = i
                    
            winner_id = (leader_id + winner_rel_idx) % 4
            partner_id = (state.my_id + 2) % 4
            we_winning = (winner_id == partner_id)
            
            best_score = -99999
            best_idx = 0
            
            for idx, move in enumerate(action_options):
                score = 0
                card = move[0]
                suit = state.get_suit(card)
                w = state.get_rank_weight(card, GlobalLevel, GlobalMajorSuit, GlobalPointOrder)
                points = sum(self.get_point_value(c) for c in move)
                
                can_beat = False
                if suit == leader_suit and not is_ruffed:
                    if w > winning_weight: can_beat = True
                elif suit == 'master' and leader_suit != 'master':
                    if not is_ruffed: can_beat = True
                    elif w > winning_weight: can_beat = True
                
                if we_winning:
                    score += points * 50 
                    score -= w 
                else:
                    if can_beat:
                        score += 1000 
                        score -= w 
                    else:
                        if points > 0: score -= 1000 
                        score -= w 
                        if suit != 'master' and suit != leader_suit:
                             count_in_hand = sum(1 for c in hold_names if state.get_suit(c) == suit)
                             if count_in_hand == len(move): score += 100
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
            return best_idx
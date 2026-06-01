import json
import os
import sys
import argparse
import torch
import numpy as np
from collections import Counter

from tractorbot.config import load_config
from tractorbot.models.model import CNNModel
from tractorbot.envs.wrapper import cardWrapper
from tractorbot.envs.mvGen import move_generator
from tractorbot.agents.heu_botzone_bot import HeuBot

class NeuralBrain:
    def __init__(self, config):
        play_config = config.get("play", {})
        self.wrapper = cardWrapper()
        self.model = CNNModel()
        self.device = play_config.get("device", "cpu")
        self.loaded = False
        
        model_path = play_config.get("model_path", "/data/model_plus_global.pt")
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.loaded = True
            except Exception as e:
                sys.stderr.write(f"[NN Load Error] {e}\n")
        else:
            sys.stderr.write(f"[NN Warning] File not found: {model_path}\n")

    def get_suggestion(self, requests, hold, self_id, banker_pos, level, major, action_options_names):
        if not self.loaded: return None
        try:
            # Rebuild the visible played-card history.
            all_played = []
            current_round_history = []
            for req in requests:
                if req["stage"] == "play":
                    hist = req["history"]
                    if hist[0]:
                        for move in hist[0]:
                            for cid in move: all_played.append(Num2Poker(cid))
                    current_round_history = hist[1]
            if current_round_history:
                for move in current_round_history:
                    for cid in move: all_played.append(Num2Poker(cid))

            deck_names = [Num2Poker(c) for c in hold]
            hist_names = [[Num2Poker(c) for c in move] for move in current_round_history]
            major_list = ['jo', 'Jo']
            if major != 'n': major_list.append(major + level)
            
            played_formatted = [all_played, [], [], []]
            
            obs = {
                "id": self_id,
                "deck": deck_names,
                "history": hist_names,
                "major": major_list,
                "played": played_formatted,
                "banker_pos": banker_pos
            }
            
            wrapped = self.wrapper.obsWrap(obs, action_options_names)
            state_input = {
                'observation': torch.tensor(wrapped['observation'], dtype=torch.float).unsqueeze(0),
                'global_feature': torch.tensor(wrapped['global_feature'], dtype=torch.float).unsqueeze(0),
                'action_mask': torch.tensor(wrapped['action_mask'], dtype=torch.float).unsqueeze(0)
            }
            
            with torch.no_grad():
                logits, _ = self.model(state_input)
                masked_logits = logits.masked_fill(state_input['action_mask'] == 0, -1e9)
                action_idx = torch.argmax(masked_logits).item()
            return action_idx
        except Exception as e:
            sys.stderr.write(f"[NN Inference Error] {e}\n")
            return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run the TractorBot play entry point.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. Defaults to configs/default.yaml or TRACTORBOT_CONFIG.",
    )
    return parser.parse_known_args()[0]

cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
suitset = ['s','h','c','d']
Major = ['jo', 'Jo']
pointorder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']

def setMajor(major, level):
    global Major, pointorder
    local_pointorder = [p for p in pointorder if p != level]
    
    if major != 'n': 
        Major = [major+point for point in local_pointorder] + \
                [suit + level for suit in suitset if suit != major] + \
                [major + level] + \
                ['jo', 'Jo']
    else: 
        Major = [suit + level for suit in suitset] + ['jo', 'Jo']
    
    if level in pointorder:
        pointorder.remove(level)
    
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

def checkPokerType(poker, level): 
    poker = [Num2Poker(p) for p in poker]
    if len(poker) == 1: return "single"
    if len(poker) == 2:
        if poker[0] == poker[1]: return "pair"
        else: return "suspect"
    if len(poker) % 2 == 0: 
        count = Counter(poker)
        if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2:
            return "tractor"
        elif "jo" in count.keys() or "Jo" in count.keys(): return "suspect"
        for v in count.values(): 
            if v != 2: return "suspect"
        return "tractor" 
    return "suspect"

def get_action_options(deck, history, level, mv_gen):
    deck_names = [Num2Poker(p) for p in deck]
    if len(history) == 4 or len(history) == 0: 
        return mv_gen.gen_all(deck_names)
    else:
        tgt = [Num2Poker(p) for p in history[0]]
        poktype = checkPokerType(history[0], level)
        if poktype == "single": return mv_gen.gen_single(deck_names, tgt)
        elif poktype == "pair": return mv_gen.gen_pair(deck_names, tgt)
        elif poktype == "tractor": return mv_gen.gen_tractor(deck_names, tgt)
        return mv_gen.gen_throw(deck_names, tgt) 

def call_Snatch(get_card, deck, called, snatched, level):
    response = []
    if snatched != -1: return []
    current_hand = deck + [get_card]
    
    def find_pair_ids(names, target):
        ids = [pid for pid, name in zip(current_hand, names) if name == target]
        return ids[:2] if len(ids)>=2 else []
    
    hand_names = [Num2Poker(i) for i in current_hand]
    
    if called == -1:
        get_poker = Num2Poker(get_card)
        if len(get_poker) > 1 and get_poker[1] == level:
            suit = get_poker[0]
            count = sum(1 for name in hand_names if name[0] == suit and name not in ['jo', 'Jo'])
            threshold = 2 if level in ['A', 'K'] else 3
            if count >= threshold: return [get_card]
        return []
    else:
        pair_Jo = find_pair_ids(hand_names, "Jo")
        if pair_Jo: return pair_Jo 
        pair_jo = find_pair_ids(hand_names, "jo")
        if pair_jo: return pair_jo
        for suit in suitset:
            pair_level = find_pair_ids(hand_names, suit + level)
            if pair_level: return pair_level
    return response

def cover_Pub(old_public, deck):
    full_hand_ids = old_public + deck
    full_hand_names = [Num2Poker(pid) for pid in full_hand_ids]
    name_counts = Counter(full_hand_names)
    trumps = []; side_suits = {s: [] for s in suitset}
    
    for pid in full_hand_ids:
        pname = Num2Poker(pid)
        if pname in Major: trumps.append(pid)
        else: side_suits[pname[0]].append(pid)
    
    to_bury = []
    sorted_suits = sorted(side_suits.keys(), key=lambda k: len(side_suits[k]))
    remaining = []
    
    for suit in sorted_suits:
        cards = side_suits[suit]
        if not cards: continue
        has_A = any(Num2Poker(c)[1] == 'A' for c in cards)
        if len(to_bury) + len(cards) <= 8 and not has_A:
            to_bury.extend(cards)
        else:
            remaining.extend(cards)
            
    def get_score(pid):
        pname = Num2Poker(pid)
        rank = pname[1]
        score = 0
        if rank == 'A': score += 100 
        elif rank in ['K', 'Q', 'J']: score += 50
        elif rank in ['5', '0', 'K']: score += 40 
        try: score += pointorder.index(rank)
        except: pass
        return score

    singles = []; pairs = []
    grouped = {}
    for pid in remaining:
        pname = Num2Poker(pid)
        if pname not in grouped: grouped[pname] = []
        grouped[pname].append(pid)
    for pids in grouped.values():
        if len(pids) == 2: pairs.append(pids)
        else: singles.extend(pids)
    singles.sort(key=get_score)
    pairs.sort(key=lambda p: get_score(p[0]))
    
    needed = 8 - len(to_bury)
    if needed > 0:
        take = min(len(singles), needed)
        to_bury.extend(singles[:take])
        needed -= take
    if needed > 0:
        for pair in pairs:
            if needed >= 2: to_bury.extend(pair); needed -= 2
            elif needed == 1: to_bury.append(pair[0]); needed -= 1; break
            else: break
    if needed > 0:
        def get_trump_score(pid):
            pname = Num2Poker(pid)
            base = Major.index(pname) if pname in Major else 0
            if name_counts[pname] >= 2: base += 10000
            return base
        trumps.sort(key=get_trump_score)
        to_bury.extend(trumps[:needed])
    return to_bury

if __name__ == '__main__':
    cfg = load_config(parse_args().config)
    brain = NeuralBrain(cfg)
    play_cfg = cfg.get("play", {})
    _online = os.environ.get("USER", "") == "root"
    if _online:
        try: full_input = json.loads(input())
        except: exit(0)
    else:
        local_input_path = play_cfg.get("local_input_path", "log_forAI.json")
        if os.path.exists(local_input_path):
            with open(local_input_path) as fo: full_input = json.load(fo)
        else:
             full_input = {"requests": [], "responses": []}

    hold = []
    requests = full_input["requests"]
    responses = full_input["responses"]
    
    # Restore local hand state from previous requests and responses.
    for i in range(len(requests) - 1):
        req = requests[i]
        
        if req["stage"] == "deal":
            hold.extend(req["deliver"])
        elif req["stage"] == "cover":
            hold.extend(req["deliver"])
            if i < len(responses):
                action_cover = responses[i]
                for id in action_cover:
                    if id in hold: hold.remove(id)
        
        elif req["stage"] == "play":
            history = req["history"]
            selfid = (history[3] + len(history[1])) % 4
            if len(history[0]) != 0:
                self_move = history[0][(selfid-history[2]) % 4]
                for id in self_move:
                    if id in hold: hold.remove(id)

    curr_request = requests[-1]
    response = []

    if curr_request["stage"] == "deal":
        setMajor('n', curr_request["global"]["level"])
        response = call_Snatch(curr_request["deliver"][0], hold, 
                               curr_request["global"]["banking"]["called"], 
                               curr_request["global"]["banking"]["snatched"], 
                               curr_request["global"]["level"])

    elif curr_request["stage"] == "cover":
        level = curr_request["global"]["level"]
        major = curr_request["global"]["banking"]["major"]
        setMajor(major, level)
        response = cover_Pub(curr_request["deliver"], hold)

    elif curr_request["stage"] == "play":
        level = curr_request["global"]["level"]
        major = curr_request["global"]["banking"]["major"]
        banker_id = curr_request["global"]["banking"]["banker"]
        
        setMajor(major, level)
        mv_gen = move_generator(level, major)
        history = curr_request["history"]
        history_curr = history[1]
        selfid = (history[3] + len(history_curr)) % 4
        
        # Synchronize hand state before selecting a play action.
        if len(history[0]) != 0:
            self_move = history[0][(selfid-history[2]) % 4]
            for id in self_move:
                if id in hold: hold.remove(id)
        
        action_options = get_action_options(hold, history_curr, level, mv_gen)
        

        bot = HeuBot(level)
        heu_idx = bot.step(requests, hold, selfid, history_curr, action_options, level, major)
        
        nn_idx = brain.get_suggestion(requests, hold, selfid, banker_id, level, major, action_options)
        if nn_idx is None: nn_idx = heu_idx
        
        final_idx = heu_idx
        is_leading = (len(history_curr) == 0)
        is_banker_side = (selfid - banker_id) % 2 == 0
        
        if is_leading:
            if not is_banker_side:
                final_idx = nn_idx 
            else:
                nn_move = action_options[nn_idx]
                heu_move = action_options[heu_idx]
                
                def is_trump_card(card_name):
                    if card_name in ['jo', 'Jo']: return True
                    if card_name[1] == level: return True
                    if major != 'n' and card_name[0] == major: return True
                    return False
                
                heu_is_trump = is_trump_card(heu_move[0])
                nn_is_trump = is_trump_card(nn_move[0])
                
                if len(nn_move) > len(heu_move): 
                    final_idx = heu_idx
                    
                elif len(heu_move) > len(nn_move):
                    final_idx = heu_idx
                    
                elif any(x in ['jo','Jo'] or x[1] == level for x in heu_move) and \
                     not any(x in ['jo','Jo'] or x[1] == level for x in nn_move):
                    final_idx = heu_idx
                
                elif heu_is_trump and not nn_is_trump:
                    final_idx = heu_idx
                    
                else:
                    final_idx = nn_idx
        else:
            final_idx = heu_idx 
            
        response = map_action_to_hold_ids(action_options[final_idx], hold)

    print(json.dumps({"response": response}))

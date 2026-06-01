import random
from collections import Counter
from tractorbot.envs.mvGen import move_generator

class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
        
    def __str__(self):
        return self.ErrorInfo


class TractorEnv():
    def __init__(self, config={}):
        self.config = config or {}
        self.reward_config = self.config.get('rewards', {})
        if 'seed' in config:
            self.seed = config['seed']
        else:
            self.seed = None

        self.suit_set = ['s','h','c','d']
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.major = None
        self.level = None
        self.agent_names = ['player_%d' % i for i in range(4)]
        
    def _optimize_banker_hand(self):
        # Use the same burying heuristic as the online play entry point.
        banker_idx = self.banker_pos
        full_hand_ids = self.player_decks[banker_idx] + self.covered_card
        
        trumps = []
        side_suits = {s: [] for s in self.suit_set}
        
        for pid in full_hand_ids:
            pname = self._id2name(pid)
            if pname in self.Major:
                trumps.append(pid)
            else:
                suit = pname[0]
                if suit in side_suits:
                    side_suits[suit].append(pid)
        
        to_bury = []
        
        sorted_suits = sorted(side_suits.keys(), key=lambda k: len(side_suits[k]))
        remaining_candidates = [] 
        for suit in sorted_suits:
            cards = side_suits[suit]
            if len(cards) == 0: continue
            
            has_Ace = False
            for pid in cards:
                if self._id2name(pid)[1] == 'A':
                    has_Ace = True
                    break
            
            if len(to_bury) + len(cards) <= 8 and not has_Ace:
                to_bury.extend(cards)
            else:
                remaining_candidates.extend(cards)
        
        def get_score(pid):
            pname = self._id2name(pid)
            rank = pname[1]
            score = 0
            if rank == 'A': score += 100 
            elif rank in ['K', 'Q', 'J']: score += 50
            
            if rank in ['5', '0', 'K']: score += 40 
            else:
                try: score += self.point_order.index(rank)
                except: pass 
            return score

        singles = []
        pairs = []
        
        grouped = {}
        for pid in remaining_candidates:
            pname = self._id2name(pid)
            if pname not in grouped: grouped[pname] = []
            grouped[pname].append(pid)
            
        for pname, pids in grouped.items():
            if len(pids) == 2:
                pairs.append(pids)
            else:
                singles.extend(pids)
        
        singles.sort(key=get_score)
        pairs.sort(key=lambda p: get_score(p[0]))
        needed = 8 - len(to_bury)
        
        if needed > 0:
            take_singles = min(len(singles), needed)
            to_bury.extend(singles[:take_singles])
            needed -= take_singles
            
        if needed > 0:
            for pair in pairs:
                if needed >= 2:
                    to_bury.extend(pair)
                    needed -= 2
                elif needed == 1:
                    to_bury.append(pair[0])
                    needed -= 1
                    break
                else:
                    break
        
        if needed > 0:
            full_hand_names = [self._id2name(pid) for pid in full_hand_ids]
            name_counts = Counter(full_hand_names)
            
            def get_trump_score(pid):
                pname = self._id2name(pid)
                try: base = self.Major.index(pname)
                except: base = 0
                if name_counts[pname] >= 2: base += 10000
                return base
                
            trumps.sort(key=get_trump_score)
            to_bury.extend(trumps[:needed])
        
        self.covered_card = to_bury
        
        new_hand = []
        to_bury_copy = to_bury.copy()
        
        for pid in full_hand_ids:
            if pid in to_bury_copy:
                to_bury_copy.remove(pid)
            else:
                new_hand.append(pid)
                
        self.player_decks[banker_idx] = new_hand
          
    def reset(self, level='2', banker_pos=0, major='s'):
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        self.level = level
        self.first_round = True
        self.banker_pos = banker_pos
        if major == 'r':
            self.major = random.sample(self.suit_set, 1)[0]
        else:
            self.major = major
        self.total_deck = [i for i in range(108)] 
        random.shuffle(self.total_deck)
        self.covered_card = self.total_deck[100:]
        self.card_todeal = self.total_deck[:100]
        self.player_decks = [[] for _ in range(4)]
        self.player_decks[0] = self.card_todeal[:25]
        self.player_decks[1] = self.card_todeal[25:50]
        self.player_decks[2] = self.card_todeal[50:75]
        self.player_decks[3] = self.card_todeal[75:100]
        self._setMajor()
        self._optimize_banker_hand()
        self.mv_gen = move_generator(self.level, self.major)
        self.score = 0
        self.history = []
        self.played_cards = [[] for _ in range(4)]
        self.reward = None
        self.done = False
        self.round = 0
        
        self.round += 1
        return self._get_obs(self.banker_pos), self._get_action_options(self.banker_pos)

    
    def step(self, response):
        self.reward = None
        curr_player = response['player']
        action = response['action']
        real_action = self._checkLegalMove(action, curr_player)
        real_action = self._name2id_seq(real_action, self.player_decks[curr_player])
        self._play(curr_player, real_action)
        next_player = (curr_player + 1) % 4
        if len(self.history) == 4:
            winner = self._checkWinner(curr_player)
            next_player = winner
            if len(self.player_decks[0]) == 0: 
                self._reveal(curr_player, winner)
                self.done = True
                
                farmer_score = self.score
                win_score = self.config.get('rules', {}).get('win_score', 80)
                farmer_win = farmer_score >= win_score
                
                GAME_OVER_BONUS = self.reward_config.get('game_over_bonus', 10.0)
                
                score_margin_divisor = self.reward_config.get('score_margin_divisor', 40.0)
                score_margin = (farmer_score - win_score) / score_margin_divisor
                
                aggression_bonus = 0.0
                if farmer_win:
                    if farmer_score >= self.reward_config.get('aggression_threshold_1', 100):
                        aggression_bonus = self.reward_config.get('aggression_bonus_1', 3.0)
                    if farmer_score >= self.reward_config.get('aggression_threshold_2', 120):
                        aggression_bonus = self.reward_config.get('aggression_bonus_2', 6.0)
                
                for i in range(4):
                    agent_name = self.agent_names[i]
                    is_farmer = (i - self.banker_pos) % 2 != 0
                    
                    final_r = 0.0
                    
                    if farmer_win:
                        if is_farmer: 
                            final_r = GAME_OVER_BONUS + max(0, score_margin) * 2.0 + aggression_bonus
                        else:         
                            final_r = -GAME_OVER_BONUS - max(0, score_margin) * 2.0 - aggression_bonus
                    else:
                        if is_farmer:
                            final_r = -GAME_OVER_BONUS + min(0, score_margin) * 2.0
                        else:
                            final_r = GAME_OVER_BONUS - min(0, score_margin) * 2.0
                    
                    if self.reward is None: self.reward = {}
                    if agent_name not in self.reward: self.reward[agent_name] = 0
                    self.reward[agent_name] += final_r
        
        self.round += 1
        
        if self.reward:
            return self._get_obs(next_player), self._get_action_options(next_player), self.reward, self.done
        return self._get_obs(next_player), self._get_action_options(next_player), None, self.done
        
    
    def _raise_error(self, player, info):
        raise Error("Player_"+str(player)+": "+info)
        
    def _get_obs(self, player):
        obs = {
            "id": player,
            "deck": [self._id2name(p) for p in self.player_decks[player]],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards]
        }
        return obs

    def _get_action_options(self, player):
        deck = [self._id2name(p) for p in self.player_decks[player]]
        if len(self.history) == 4 or len(self.history) == 0:
            return self.mv_gen.gen_all(deck)
        else:
            tgt = [self._id2name(p) for p in self.history[0]]
            poktype = self._checkPokerType(self.history[0], (player-len(self.history))%4)
            if poktype == "single":
                return self.mv_gen.gen_single(deck, tgt)
            elif poktype == "pair":
                return self.mv_gen.gen_pair(deck, tgt)
            elif poktype == "tractor":
                return self.mv_gen.gen_tractor(deck, tgt)
            elif poktype == "suspect":
                return self.mv_gen.gen_throw(deck, tgt)    
    
    def _done(self):
        return self.done    
    
    def _id2name(self, card_id):
        NumInDeck = card_id % 54
        if NumInDeck == 52:
            return "jo"
        if NumInDeck == 53:
            return "Jo"
        pokernumber = self.card_scale[NumInDeck // 4]
        pokersuit = self.suit_set[NumInDeck % 4]
        return pokersuit + pokernumber
    
    def _name2id(self, card_name, deck):
        NumInDeck = -1
        if card_name[0] == "j":
            NumInDeck = 52
        elif card_name[0] == "J":
            NumInDeck = 53
        else:
            NumInDeck = self.card_scale.index(card_name[1])*4 + self.suit_set.index(card_name[0])
        if NumInDeck in deck:
            return NumInDeck
        else:
            return NumInDeck + 54
    
    def _name2id_seq(self, card_names, deck):
        id_seq = []
        deck_copy = deck + []
        for card_name in card_names:
            card_id = self._name2id(card_name, deck_copy)
            id_seq.append(card_id)
            deck_copy.remove(card_id)
        return id_seq
        
    
    def _play(self, player, cards):
        for card in cards:
            self.player_decks[player].remove(card)
            self.played_cards[player].append(card)
        if len(self.history) == 4:
            self.history = []
        self.history.append(cards)
            
    def _reveal(self, currplayer, winner):
        if self._checkPokerType(self.history[0], (currplayer-3)%4) != "suspect":
            mult = len(self.history[0])
        else:
            divided, _ = self._checkThrow(self.history[0], (currplayer-3)%4, check=False)
            divided.sort(key=lambda x: len(x), reverse=True)
            if len(divided[0]) >= 4:
                mult = len(divided[0]) * 2
            elif len(divided[0]) == 2:
                mult = 4
            else: 
                mult = 2

        publicscore = 0
        for pok in self.covered_card: 
            p = self._id2name(pok)
            if p[1] == "5":
                publicscore += 5
            elif p[1] == "0" or p[1] == "K":
                publicscore += 10
                    
        self._reward(winner, publicscore*mult, 0, 0, winner)
    
    def _setMajor(self):
        if self.major != 'n':
            self.Major = [self.major+point for point in self.point_order if point != self.level] + [suit + self.level for suit in self.suit_set if suit != self.major] + [self.major + self.level] + self.Major
        else:
            self.Major = [suit + self.level for suit in self.suit_set] + self.Major
        self.point_order.remove(self.level)
        
    def _checkPokerType(self, poker, currplayer):
        level = self.level
        poker = [self._id2name(p) for p in poker]
        if len(poker) == 1:
            return "single" 
        if len(poker) == 2:
            if poker[0] == poker[1]:
                return "pair"
            else:
                return "suspect"
        if len(poker) % 2 == 0:
            count = Counter(poker)
            if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2 and len(poker) == 4:
                return "tractor"
            elif "jo" in count.keys() or "Jo" in count.keys():
                return "suspect"
            for v in count.values():
                if v != 2:
                    return "suspect"
            pointpos = []
            suit = list(count.keys())[0][0]
            for k in count.keys():
                if k[0] != suit or k[1] == level:
                    return "suspect"
                pointpos.append(self.point_order.index(k[1]))
            pointpos.sort()
            for i in range(len(pointpos)-1):
                if pointpos[i+1] - pointpos[i] != 1:
                    return "suspect"
            return "tractor"
        
        return "suspect"

    def _checkBigger(self, poker, currplayer):
        own = self.player_decks
        level = self.level
        major = self.major
        tyPoker = self._checkPokerType(poker, currplayer)
        poker = [self._id2name(p) for p in poker]
        assert tyPoker != "suspect", "Type 'throw' should contain common types"
        own_pok = [[self._id2name(num) for num in hold] for hold in own]
        if poker[0] in self.Major:
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                major_pok = [pok for pok in hold if pok in self.Major]
                count = Counter(major_pok)
                if len(poker) <= 2:
                    if poker[0][1] == level and poker[0][0] != major:
                        if major == 'n':
                            for k,v in count.items(): 
                                if (k == 'jo' or k == 'Jo') and v >= len(poker):
                                    return True
                        else:
                            for k,v in count.items():
                                if (k == 'jo' or k == 'Jo' or k == major + level) and v >= len(poker):
                                    return True
                    else: 
                        for k,v in count.items():
                            if self.Major.index(k) > self.Major.index(poker[0]) and v >= len(poker):
                                return True
                else:
                    if "jo" in poker:
                        return False
                    if len(poker) == 4 and "jo" in count.keys() and "Jo" in count.keys():
                        if count["jo"] == 2 and count["Jo"] == 2:
                            return True
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if k != 'jo' and k != 'Jo' and k[1] != level and self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]):
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        else:
            suit = poker[0][0]
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                suit_pok = [pok for pok in hold if pok[0] == suit and pok[1] != level]
                count = Counter(suit_pok)
                if len(poker) <= 2:
                    for k, v in count.items():
                        if self.point_order.index(k[1]) > self.point_order.index(poker[0][1]) and v >= len(poker):
                            return True
                else:
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]):
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False

        return False

    def _checkThrow(self, poker, currplayer, check=False):
        """Validate and decompose a throw move."""
        own = self.player_decks
        level = self.level
        major = self.major
        ilcnt = 0
        pok = [self._id2name(p) for p in poker]
        outpok = []
        failpok = []
        count = Counter(pok)
        if check:
            if list(count.keys())[0] in self.Major:
                for p in count.keys():
                    if p not in self.Major:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
            else:
                suit = list(count.keys())[0][0]
                for k in count.keys():
                    if k[0] != suit:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
        # Extract tractors before processing simple pairs and singles.
        pos = []
        tractor = []
        suit = ''
        for k, v in count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level:
                    pos.append(self.point_order.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos)-1):
                if pos[i+1]-pos[i] == 1:
                    if not suc_flag:
                        tmp = [suit + self.point_order[pos[i]], suit + self.point_order[pos[i]], suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]]
                        del count[suit + self.point_order[pos[i]]]
                        del count[suit + self.point_order[pos[i+1]]]
                        suc_flag = True
                    else:
                        tmp.extend([suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]])
                        del count[suit + self.point_order[pos[i+1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        for k,v in count.items(): 
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)

        if check:
            for poktype in outpok:
                if self._checkBigger(poktype, currplayer):
                    ilcnt += len(poktype)
                    failpok.append(poktype)  
        
        if ilcnt > 0:
            finalpok = []
            kmin = ""
            for poktype in failpok:
                getmark = poktype[-1] 
                if kmin == "":
                    finalpok = poktype
                    kmin = getmark
                elif kmin in self.Major:
                    if self.Major.index(getmark) < self.Major.index(kmin):
                        finalpok = poktype
                        kmin = getmark
                else:
                    if self.point_order.index(getmark[1]) < self.point_order.index(kmin[1]):
                        finalpok = poktype
                        kmin = getmark
            finalpok = [[finalpok[0]]]
        else: 
            finalpok = outpok

        return finalpok, ilcnt 
        
        
    def _checkRes(self, poker, own):
        level = self.level
        pok = [self._id2name(p) for p in poker]
        own_pok = [self._id2name(p) for p in own]
        if pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            count = Counter(major_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        if k != 'jo' and k != 'Jo' and k[1] != level:
                            pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        else:
            suit = pok[0][0]
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            count = Counter(suit_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        return False
    
    def _checkLegalMove(self, poker, currplayer):
        """Return the normalized move or raise when a move violates follow-suit rules."""
        level = self.level
        major = self.major
        own = self.player_decks
        banker = self.banker_pos
        history = self.history
        pok = [self._id2name(p) for p in poker]
        hist = [[self._id2name(p) for p in move] for move in history]
        outpok = pok
        own_pok = [self._id2name(p) for p in own[currplayer]]
        if len(history) == 0 or len(history) == 4:
            typoker = self._checkPokerType(poker, currplayer)
            if typoker == "suspect":
                outpok_s, ilcnt = self._checkThrow(poker, currplayer, True)
                if ilcnt > 0:
                    self._punish(currplayer, ilcnt*10)
                outpok = [p for poktype in outpok_s for p in poktype]
        else:
            tyfirst = self._checkPokerType(history[0], currplayer)
            if len(poker) != len(history[0]):
                self._raise_error(currplayer, "ILLEGAL_MOVE")
            if tyfirst == "suspect":
                outhis, ilcnt = self._checkThrow(history[0], currplayer, check=False)
                # A previously accepted throw is only checked for legal following.
                flathis = [p for poktype in outhis for p in poktype]
                if outhis[0][0] in self.Major: 
                    major_pok = [p for p in pok if p in self.Major]
                    if len(major_pok) != len(poker):
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(major_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else:
                        outhis.sort(key=lambda x: len(x), reverse=True)
                        major_hold = [p for p in own_pok if p in self.Major]
                        matching = True
                        if self._checkPokerType(outhis[0], currplayer) == "tractor":
                            divider, _ = self._checkThrow(poker, currplayer, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(major_hold, currplayer, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis:
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching:
                                res_ex = True
                                for chtype in own_cnt:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex:
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else:
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                else:
                    suit = hist[0][0][0]
                    suit_pok = [p for p in pok if p not in self.Major and p[0] == suit]
                    if len(suit_pok) != len(poker):
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        if len(suit_pok) != len(suit_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else: 
                        outhis.sort(key=lambda x: len(x), reverse=True)
                        suit_hold = [p for p in own_pok if p not in self.Major and p[0] == suit]
                        matching = True
                        if self._checkPokerType(outhis[0], currplayer) == "tractor":
                            divider, _ = self._checkThrow(poker, currplayer, check=False)
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(suit_hold, currplayer, check=False)
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for poktype in outhis:
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching:
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else: 
                                        res_ex = False
                                        break
                                if res_ex:
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else:
                                    pair_own = sum([len(x) for x in own_divide if len(x) >= 2])
                                    pair_his = sum([len(x) for x in outhis if len(x) >= 2])
                                    pair_pok = sum([len(x) for x in divider if len(x) >= 2])
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")

            else:
                if self._checkRes(history[0], own[currplayer]):
                    if self._checkPokerType(poker, currplayer) != tyfirst:
                        self._raise_error(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] in self.Major and pok[0] not in self.Major:
                        self._raise_error(currplayer,"ILLEGAL_MOVE")
                    if hist[0][0] not in self.Major and (pok[0] in self.Major or pok[0][0] != hist[0][0][0]):
                        self._raise_error(currplayer, "ILLEGAL_MOVE") 
                elif self._checkPokerType(poker, currplayer) != tyfirst:
                    own_pok = [self._id2name(p) for p in own[currplayer]]
                    if hist[0][0] in self.Major:
                        major_pok = [p for p in pok if p in self.Major]
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(poker):
                            if len(major_pok) != len(major_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else:
                            count = Counter(major_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0])/2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs:
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")

                    else: 
                        suit = hist[0][0][0]
                        suit_pok = [p for p in pok if p[0] == suit and p not in self.Major]
                        suit_hold = [p for p in own_pok if p[0] == suit and p not in self.Major]
                        if len(suit_pok) != len(poker):    
                            if len(suit_pok) != len(suit_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else:
                            count = Counter(suit_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0])/2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if pkpairs < trpairs and pkpairs < hdpairs:
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                        
        return outpok
    
    def _checkWinner(self, currplayer):
        level = self.level
        major = self.major
        history = self.history
        histo = history + []
        hist = [[self._id2name(p) for p in x] for x in histo]
        score = 0 

        _kill = 0
        _tractor = 0   
        _throw = 0
        _double = 0    
        first_player = (currplayer - 3) % 4

        for move in hist:
            for pok in move:
                if pok[1] == "5":
                    score += 5
                elif pok[1] == "0" or pok[1] == "K":
                    score += 10
        
        win_seq = 0
        win_move = hist[0]
        tyfirst = self._checkPokerType(history[0], currplayer)
        
        if tyfirst == "suspect":
            first_parse, _ = self._checkThrow(history[0], currplayer, check=False)
            first_parse.sort(key=lambda x: len(x), reverse=True)
            for i in range(1,4):
                move_parse, r = self._checkThrow(history[i], currplayer, check=False)
                move_parse.sort(key=lambda x: len(x), reverse=True)
                move_cnt = [len(x) for x in move_parse]
                matching = True
                for poktype in first_parse:
                    if move_cnt[0] >= len(poktype):
                        move_cnt[0] -= len(poktype)
                        move_cnt.sort(reverse=True)
                    else:
                        matching = False
                        break
                if not matching:
                    continue
                if hist[i][0] not in self.Major:
                    continue
                if win_move[0] not in self.Major and hist[i][0] in self.Major:
                    win_move = hist[i]
                    win_seq = i
                    _kill = 1
                elif len(first_parse[0]) >= 4:
                    if major == 'n':
                        continue
                    win_parse, s = self._checkThrow(history[win_seq], currplayer, check=False)
                    win_parse.sort(key=lambda x: len(x), reverse=True)
                    if self.Major.index(win_parse[0][-1]) < self.Major.index(move_parse[0][-1]):
                        win_move = hist[i]
                        win_seq = i
                        _throw = 1
                else: 
                    step = len(first_parse[0])
                    win_count = Counter(win_move)
                    win_max = 0
                    for k,v in win_count.items():
                        if v >= step and self.Major.index(k) >= win_max: 
                            win_max = self.Major.index(k)
                    move_count = Counter(hist[i])
                    move_max = 0
                    for k,v in move_count.items():
                        if v >= step and self.Major.index(k) >= move_max:
                            move_max = self.Major.index(k)
                    if major == 'n':
                        if self.Major[win_max][1] == level:
                            if self.Major[move_max] == 'jo' or self.Major[move_max] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(move_max) > self.Major.index(win_max):
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major[win_max][1] == level and self.Major[win_max][0] != major:
                        if (self.Major[move_max][0] == major and self.Major[move_max][1] == level) or self.Major[move_max] == "jo" or self.Major[move_max] == "Jo":
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major.index(win_max) < self.Major.index(move_max):
                        win_move = hist[i]
                        win_seq = i

        else:
            for i in range(1, 4):
                if self._checkPokerType(history[i], currplayer) != tyfirst:
                    continue
                if (hist[0][0] in self.Major and hist[i][0] not in self.Major) or (hist[0][0] not in self.Major and (hist[i][0] not in self.Major and hist[i][0][0] != hist[0][0][0])):
                    continue
                elif win_move[0] in self.Major:
                    if hist[i][0] not in self.Major:
                        continue
                    if major == 'n':
                        if win_move[-1][1] == level:
                            if hist[i][-1] == 'jo' or hist[i][-1] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                    else:
                        if win_move[-1][0] != major and win_move[-1][1] == level:
                            if (hist[i][-1][0] == major and hist[i][-1][1] == level) or hist[i][-1] == 'jo' or hist[i][-1] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                else:
                    if hist[i][0] in self.Major:
                        win_move = hist[i]
                        win_seq = i
                        _kill = 1
                    elif self.point_order.index(win_move[0][-1]) < self.point_order.index(hist[i][0][-1]):
                        win_move = hist[i]
                        win_seq = i
        
        win_id = (currplayer - 3 + win_seq) % 4

        self._reward(win_id, score, _kill, _throw, first_player)

        return win_id
    
    def _reward(self, player, points, kill, throw, first):
        if (player - self.banker_pos) % 2 != 0: 
            self.score += points
            
        history_map = {}
        for i in range(4):
            rel_idx = (i - first) % 4
            try:
                raw_cards = self.history[rel_idx]
                history_map[i] = [self._id2name(c) for c in raw_cards]
            except:
                history_map[i] = []
            
        self.reward = {name: 0.0 for name in self.agent_names}
        
        WIN_TRICK_REWARD = self.reward_config.get('win_trick', 0.5)
        KILL_REWARD = self.reward_config.get('kill', 0.5)
        THROW_REWARD_BASE = self.reward_config.get('throw_base', 0.5)
        THROW_EXTRA = self.reward_config.get('throw_extra', 0.1)
        POINT_UNIT = self.reward_config.get('point_unit', 50.0)
        
        BANKER_PUNISH_MULTIPLIER = self.reward_config.get('banker_punish_mult', 1.5)
            
        for i in range(4):
            agent_name = self.agent_names[i]
            is_partner = (i - player) % 2 == 0
            is_banker_side = (i - self.banker_pos) % 2 == 0 
            
            if i == player and throw:
                self.reward[agent_name] += THROW_REWARD_BASE
                win_cards = history_map[player]
                if len(win_cards) > 2:
                    self.reward[agent_name] += (len(win_cards) - 2) * THROW_EXTRA
            
            if i == first:
                my_cards = history_map[i]
                length = len(my_cards)
                if length == 2:
                    self.reward[agent_name] += 0.3
                if length > 0 and my_cards[0][0] not in self.Major:
                    if my_cards[0][1] == 'A':
                        self.reward[agent_name] += 0.5
            
            r_base = 0.0
            r_point = 0.0
            
            if is_partner:
                if i == player: 
                    r_base += WIN_TRICK_REWARD 
                    if kill: r_base += KILL_REWARD
                else:
                    r_base += WIN_TRICK_REWARD * 0.5
            else:
                r_base -= 0.05

            if points > 0:
                base_point_val = points / POINT_UNIT
                
                if (player - self.banker_pos) % 2 == 0:
                    if is_banker_side:
                        r_point = base_point_val
                    else:
                        r_point = -base_point_val
                
                else:
                    if not is_banker_side:
                        r_point = base_point_val
                    else:
                        r_point = -base_point_val * BANKER_PUNISH_MULTIPLIER

                if is_partner and i != player:
                    r_point *= 0.5
            
            self.reward[agent_name] += r_base + r_point

    def _punish(self, player, points):
        if (player-self.banker_pos) % 2 != 0:
            self.score -= points
        else:
            self.score += points
    
    def action_intpt(self, action, player):
        '''
        interpreting action(cardname) to response(dick{'player': int, 'action': list[int]})
        action: list[str(cardnames)]
        '''
        player_deck = self.player_decks[player]
        action = self._name2id_seq(action, player_deck)
        return {'player': player, 'action': action}
        

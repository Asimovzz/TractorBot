import numpy as np
import torch

class cardWrapper:
    def __init__(self, suit_sequence=['s', 'h', 'c', 'd'], point_sequence = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']):
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.suit_sequence = suit_sequence
        self.point_sequence = point_sequence
        self.J_pos = self.suit_sequence.index('h')
        self.j_pos = self.suit_sequence.index('s')
        
    def name2pos(self, cardname):
        if cardname[0] == "J":
            return (self.J_pos, 13)
        if cardname[0] == "j":
            return (self.j_pos, 13)
        pos = (self.suit_sequence.index(cardname[0]), self.point_sequence.index(cardname[1]))
        return pos
    
    def pos2name(self, cardpos):
        if cardpos[1] == 13:
            if cardpos[0] == self.j_pos:
                return "jo"
            if cardpos[0] == self.J_pos:
                return "Jo"
            else:
                raise Exception("Card not exists.")
        
        return self.suit_sequence[cardpos[0]] + self.point_sequence[cardpos[1]]
    
    # adding cards to a cardset 
    def add_card(self, cardset: np.array, cards): 
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[0, card_pos[0], card_pos[1]] == 0:
                cardset[0, card_pos[0], card_pos[1]] = 1
            elif cardset[1, card_pos[0], card_pos[1]] == 0:
                cardset[1, card_pos[0], card_pos[1]] = 1    
            else:
                pass
        return cardset
    
    # removing cards from cardset
    def remove_card(self, cardset: np.array, cards):
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[1, card_pos[0], card_pos[1]] != 0:
                cardset[1, card_pos[0], card_pos[1]] = 0
            elif cardset[0, card_pos[0], card_pos[1]] != 0:
                cardset[0, card_pos[0], card_pos[1]] = 0
            else:
                raise Exception("Card not in cardset! Please recheck.")
        return cardset
    
    # From cardset to cardnames
    def Unwrap(self, cardset): 
        cards = []
        card_poses = np.nonzero(cardset)
        for i in range(card_poses[0].size):
            card_name = self.pos2name((card_poses[1][i], card_poses[2][i]))
            cards.append(card_name)
        return cards

    def get_global_features(self, played_mat, obs):
        """
        提取 6 维全局特征向量
        游戏进度 (0-1)
        当前已出分值
        庄家身份 (0/1)
        关键牌消耗率: A (0-1)
        关键牌消耗率: K (0-1)
        关键牌消耗率: 分牌 (0-1)
        """
        
        played_count = np.sum(played_mat)
        progress = played_count / 100.0
        
        rank_counts = np.sum(played_mat, axis=(0, 1)) # shape [14]
        
        
        count_A = rank_counts[12]
        count_K = rank_counts[11]
        count_Points = rank_counts[3] + rank_counts[8] + rank_counts[11]
        
        feat_A = count_A / 8.0
        feat_K = count_K / 8.0
        feat_Points = count_Points / 24.0 
        
        is_banker = 0.5
        if 'banker_pos' in obs:
            is_same_team = (obs['id'] - obs['banker_pos']) % 2 == 0
            is_banker = 1.0 if is_same_team else 0.0
        
        curr_score = count_Points * 10.0 / 200.0
        
        features = np.array([
            progress,
            curr_score,
            is_banker,
            feat_A,
            feat_K,
            feat_Points
        ], dtype=np.float32)
        
        return features

    def obsWrap(self, obs, options):
        '''
        Wrapping the observation and craft the action_mask
        obs: raw obs from env
        '''
        id = obs['id']
        major_mat = np.zeros((2,4,14))
        deck_mat = np.zeros((2,4,14))
        hist_mat = np.zeros((8,4,14)) 
        played_mat = np.zeros((8,4,14))
        option_mat = np.zeros((108,4,14))
        
        self.add_card(major_mat, obs['major'])
        self.add_card(deck_mat, obs['deck'])
        for i in range(len(obs['history'])):
            self.add_card(hist_mat[i*2:(i+1)*2], obs['history'][i])
            
        played_cards = obs['played'][id:]+obs['played'][:id]
        for i in range(len(played_cards)):
            self.add_card(played_mat[i*2:(i+1)*2], played_cards[i])
            
        for i in range(len(options)):
            if i*2 >= option_mat.shape[0]:
                break
            self.add_card(option_mat[i*2:(i+1)*2], options[i])
        
        action_mask = np.zeros(54)
        action_mask[:len(options)] = 1
        
        global_feat = self.get_global_features(played_mat, obs)
        
        cnn_input = np.concatenate((major_mat, deck_mat, hist_mat, played_mat, option_mat))
        
        return {
            'observation': cnn_input,
            'global_feature': global_feat,
            'action_mask': action_mask
        }
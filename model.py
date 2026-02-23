import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(20, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResBlock(64, 128),
                ResBlock(128, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        
        # 解耦
        self.actor_encoder = make_encoder()
        self.critic_encoder = make_encoder()
        
        # 全局特征处理
        self.global_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Action Encoder
        self.action_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResBlock(16, 32),
            ResBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # Output: [B*54, 64]
        )
        
        # Actor Projection
        self.state_proj = nn.Linear(320, 64)
        
        # Critic Head
        self.value_head = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

    def forward(self, input_dict):
        # input_dict["observation"] shape: [Batch, 128, 4, 14]
        # input_dict["global_feature"] shape: [Batch, 6]
        
        obs = input_dict["observation"].float()
        global_input = input_dict["global_feature"].float()
        
        # 拆分 State 和 Options
        global_state = obs[:, :20, :, :] 
        option_mat = obs[:, 20:, :, :] 
        
        batch_size = obs.shape[0]
        
        # CNN 特征
        actor_cnn_feat = self.actor_encoder(global_state)
        critic_cnn_feat = self.critic_encoder(global_state)
        
        # 全局特征
        global_emb = self.global_fc(global_input)
        
        # 特征融合
        actor_fusion = torch.cat([actor_cnn_feat, global_emb], dim=1)
        critic_fusion = torch.cat([critic_cnn_feat, global_emb], dim=1)
        
        actions_reshaped = option_mat.reshape(batch_size, 54, 2, 4, 14)
        actions_flat = actions_reshaped.reshape(-1, 2, 4, 14).contiguous()
        action_feat = self.action_conv(actions_flat)
        action_feat = action_feat.reshape(batch_size, 54, 64)
        
        state_query = self.state_proj(actor_fusion)
        state_query = state_query.unsqueeze(1)
        
        logits = torch.sum(state_query * action_feat, dim=-1) / math.sqrt(64)
        
        mask = input_dict["action_mask"].bool()
        masked_logits = logits.masked_fill(~mask, -1e9)
        
        value = self.value_head(critic_fusion)
        
        return masked_logits, value
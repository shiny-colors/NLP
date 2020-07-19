# ライブラリの読み込み
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
import random

device = "cpu"

# モデルの定義
# 埋め込み層を定義
class Embedding(nn.Module):
    def __init__(self, in_features, max_pt):
        super().__init__()
        self.theta_h = nn.Embedding(num_embeddings=max_pt, embedding_dim=in_features)
        nn.init.xavier_normal_(self.theta_h.weight)
        
    def forward(self, pt_id, max_pt, device):
        theta_h = self.theta_h(torch.arange(max_pt).to(device))
        h_features = theta_h[pt_id, ]
        return h_features

# MultiHead Attention Block層を定義
class SelfAttention(nn.Module):
    def forward(self, hidden_q, hidden_k, hidden_g, id_box, k, d_k):
        input_mask = torch.BoolTensor(id_box==k).unsqueeze(1).to(device)
        weights = torch.matmul(hidden_q, hidden_k.transpose(-2, -1)) / np.sqrt(d_k)
        mask = input_mask.unsqueeze(1)
        weights = weights.masked_fill(mask==1, -1e9)
        normalized_weights = F.softmax(weights, dim=-1)
        score = torch.matmul(normalized_weights, hidden_g)
        return score

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, in_features, out_features, dropout_prob):
        super().__init__()
        self.in_features = in_features
        self.heads = heads
        self.d_k = in_features // heads
        
        self.gamma_q = nn.Linear(in_features, in_features, bias=False)
        self.gamma_k = nn.Linear(in_features, in_features, bias=False)
        self.gamma_g = nn.Linear(in_features, in_features, bias=False)
        self.gamma_o = nn.Linear(in_features, in_features, bias=False)
        
        self.dropout = nn.Dropout(dropout_prob)
        nn.init.xavier_normal_(self.gamma_q.weight)
        nn.init.xavier_normal_(self.gamma_k.weight)
        nn.init.xavier_normal_(self.gamma_g.weight)
        nn.init.xavier_normal_(self.gamma_o.weight)
        
    def forward(self, features, id_box, k):
        
        # 全結合層で特徴量を変換
        bs = features.shape[0]
        hidden_q = self.gamma_q(features).reshape(bs, -1, self.heads, self.d_k).transpose(1, 2)
        hidden_k = self.gamma_k(features).reshape(bs, -1, self.heads, self.d_k).transpose(1, 2)
        hidden_g = self.gamma_g(features).reshape(bs, -1, self.heads, self.d_k).transpose(1, 2)
        
        # Attention Mapの特徴量を変換
        score = SelfAttention()(hidden_q, hidden_k, hidden_g, id_box, k, self.d_k)
        score = self.dropout(score)
        concat_score = score.transpose(1, 2).contiguous().view(bs, -1, self.in_features)
        concat_score = self.dropout(concat_score)
        output = self.gamma_o(concat_score)
        return output
    
    
# Transformer Block層を定義
class Transformer(nn.Module):
    def __init__(self, heads, in_features, out_features, dropout_prob):
        super().__init__()
        self.attention_model = MultiHeadAttention(heads, in_features, out_features, dropout_prob)
        self.gamma_f1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma_f2 = nn.Linear(out_features, in_features, bias=False)
        self.layernorm1 = nn.LayerNorm(in_features)
        self.layernorm2 = nn.LayerNorm(in_features)
        
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        nn.init.xavier_normal_(self.gamma_f1.weight)
        nn.init.xavier_normal_(self.gamma_f2.weight)
        
    def forward(self, features, id_box, k):
        # Self Attentionで特徴量を変換
        normalized_features = self.layernorm1(features)
        attention_features = self.attention_model(features, id_box, k)

        # 正規化とfeed forward層
        dropout_attention = features + self.dropout1(attention_features)
        normalized_attention = self.layernorm2(dropout_attention)
        features_ff1 = self.dropout2(F.relu(self.gamma_f1(normalized_attention)))
        features_ff2 = dropout_attention + self.gamma_f2(features_ff1)
        return features_ff2
    
# 行列分解層を定義
class DMF(nn.Module):
    def __init__(self, in_features, out_features, classes, C):
        super().__init__()
        self.gamma11 = nn.ModuleList([nn.Linear(in_features, out_features) for j in range(C)])
        self.gamma12 = nn.ModuleList([nn.Linear(in_features, out_features) for j in range(C)])
        self.gamma21 = nn.Linear(2*out_features, classes, bias=True)
        self.gamma22 = nn.Linear(out_features, classes, bias=False)
        
        # 重み初期化処理
        for j in range(C):
            nn.init.xavier_normal_(self.gamma11[j].weight)
            nn.init.xavier_normal_(self.gamma12[j].weight)
        nn.init.xavier_normal_(self.gamma21.weight)
        nn.init.xavier_normal_(self.gamma22.weight)
        
    def forward(self, x1, x2, distance):
        ff1 = F.relu(distance*self.gamma11[0](x1) + (1-distance)*self.gamma11[1](x1))
        ff2 = F.relu(distance*self.gamma12[0](x2) + (1-distance)*self.gamma12[1](x2))
        logit = self.gamma21(torch.cat((ff1, ff2), dim=1)) + self.gamma22(ff1 * ff2)
        return logit
    
    
# 結合層を定義
class Joint(nn.Module):
    def __init__(self, heads, in_features1, in_features2, out_features, out_dim, classes, max_pt, C, dropout_prob):
        super().__init__()
        self.gamma = nn.Linear(in_features1, in_features2)
        self.embedding_model = Embedding(in_features2, max_pt)
        self.transformer_model11 = Transformer(heads, in_features2, out_features, dropout_prob)
        self.transformer_model12 = Transformer(heads, in_features2, out_features, dropout_prob)
        
        self.transformer_model21_1 = Transformer(heads, in_features2, out_features, dropout_prob)
        self.transformer_model21_2 = Transformer(heads, in_features2, out_features, dropout_prob)
        self.transformer_model22_1 = Transformer(heads, in_features2, out_features, dropout_prob)
        self.transformer_model22_2 = Transformer(heads, in_features2, out_features, dropout_prob)
        self.dmf_model = DMF(in_features2, out_dim, classes, C)
        
    def forward(self, bert_feature, feature_phrase, distance, word_box, phrase_box, pt_id, 
                phrase_index, D, phrase, max_pt, max_m, zeros, device):
        
        reduction_feature = F.relu(self.gamma(bert_feature))
        h_features = self.embedding_model(pt_id, max_pt, device)
        features_transformer11 = self.transformer_model11(reduction_feature, word_box, 0)
        features_transformer12 = self.transformer_model12(features_transformer11, word_box, 0)
        features_tensor2 = torch.cat((features_transformer12[:, 0, :] + h_features, zeros), dim=0)[phrase_box, ]
        
        features_transformer21_1 = self.transformer_model21_1(features_tensor2, phrase_box, phrase)
        features_transformer21_2 = self.transformer_model21_2(features_transformer21_1, phrase_box, phrase)
        features_transformer22_1 = self.transformer_model22_1(features_tensor2, phrase_box, phrase)
        features_transformer22_2 = self.transformer_model22_2(features_transformer22_1, phrase_box, phrase)
        
        x1 = features_transformer21_2.reshape(D*max_m, in_features2)[phrase_index, ][feature_phrase[:, 0], ]
        x2 = features_transformer22_2.reshape(D*max_m, in_features2)[phrase_index, ][feature_phrase[:, 1], ]
        logit = self.dmf_model(x1, x2, distance)
        return logit
    
# https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/bidirectional_transformer.py#L106

import torch
from torch.nn import functional as F
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """

    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x,query):
        #Use built-in multihead attention, where x is the key, and value,
        attn, _ = self.MultiHeadAttention(query, x, x, need_weights=False)
        attn = self.dropout(attn)
        # print(attn.shape)
        out = query.add(attn)
        out = self.LayerNorm1(out)
        mlp = self.MLP(out)
        # print(mlp.shape)
        out = out.add(mlp)
        out = self.LayerNorm2(out)
        return out


class Decoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    Same thing as the encoder, but now one of the queries is an input
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Decoder, self).__init__()
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, query):
        # Use built-in multihead attention, where x is the key, and value,
        attn, _ = self.MultiHeadAttention(query, x, x, need_weights=False)
        attn = self.dropout(attn) #Dropout layer
        out = query.add(attn) #Skip connection
        out = self.LayerNorm1(out) #Layer norm
        mlp = self.MLP(out)  #MlP
        out = out.add(mlp) #Skip connection
        out = self.LayerNorm2(out) #Layer norm
        return out


class BidirectionalTransformer(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer, self).__init__()
        # Embeddings of tokens (codes) and positions ( not learned)
        # self.tok_emb = nn.Embedding(args.num_codebook_vectors + 1, args.dim,device = args.dev)
        self.CNN_kernel = 9
        self.tok_emb_CNN = nn.Sequential(*[
            nn.Conv2d(1, 50, self.CNN_kernel),
            nn.GELU(),
            nn.Conv2d(50, 100, self.CNN_kernel),
            nn.GELU(),
            # Patchwise linear output dim = 1x200x(args.psz-(self.CNN_kernel*2 -2))xtwice
            nn.Conv2d(100, 100, self.CNN_kernel),
            nn.GELU(),
            nn.Conv2d(100, 200, 1),
            nn.Conv2d(200, 200, 1)
        ])
        self.CNN_linear_dim = 200*(args.psz-(self.CNN_kernel -1)*3)**2
        self.tok_emb_CNN_Linear =nn.Linear(self.CNN_linear_dim,args.dim)


        self.pos_emb_MLP = nn.Sequential(*[
            nn.Linear(2, 2048),
            nn.GELU(),
            nn.Linear(2048, args.dim),
            nn.Dropout(p=0.1)
        ])

        # The actual transformer encoder architecture
        self.enc_blocks = nn.ModuleList([Encoder(args.dim, args.hidden_dim) for _ in range(args.n_layers)])
        self.dec_blocks = nn.ModuleList([Decoder(args.dim, args.hidden_dim) for _ in range(4)])# rn just 2 layers
        # self.dec_blocks = Decoder(args.dim, args.hidden_dim)  # 1 layer

        #Final prediction is just linear + sigmoid
        self.decoder_pred = nn.Sequential(*[nn.Linear(args.dim,args.hidden_dim,bias=True),nn.GELU(),nn.Linear(args.hidden_dim,1,bias=True),nn.Sigmoid()])

        self.ln = nn.LayerNorm(args.dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        self.apply(weights_init)
        self.psz = args.psz
        self.normalize_embed = args.normalize_embed
        self.scale_embed = args.scale_embed

    def forward(self, x, posns=None, mask_posns=None):
        token_embeddings = self.tok_emb_CNN(x.reshape(-1, 1, self.psz, self.psz))
        # CNN token embeddings
        token_embeddings = token_embeddings.reshape(-1, self.CNN_linear_dim)
        # print(token_embeddings.shape)
        token_embeddings = self.tok_emb_CNN_Linear(token_embeddings)
        position_embeddings = self.pos_emb_MLP(posns)

        if self.normalize_embed:
            token_embeddings = (token_embeddings.T / token_embeddings.abs().max(dim=1)[0]).T
            position_embeddings = (position_embeddings.T / position_embeddings.abs().max(dim=1)[0]).T

        embed = self.drop(self.ln(token_embeddings*self.scale_embed + position_embeddings))
        for enc_block in self.enc_blocks:
            embed = enc_block(embed, position_embeddings)
        # Embed the input mask positions and pass them as queries
        mask_embeds = self.pos_emb_MLP(mask_posns)
        embed = self.dec_blocks[0](embed, mask_embeds)
        for dec_block in self.dec_blocks[1:]:
            embed = dec_block(embed, embed)
        # embed = self.dec_blocks(embed,mask_embeds)
        out = self.decoder_pred(embed)
        return out

import torch
from torch import nn

class Embedding(nn.Module):
    '''
    Create embeddings.
    Input:
    EMBED_DIMS: Embedding dimension
    GLOBAL_EMBED_LENGTH: Length of the global view embedding
    LOCAL_EMBED_LENGTH: Length of the local view embedding

    Returns the embedding tensors for global and local view, as well as the stellar parameters table (not embedded).
    '''
    def __init__(self,EMBED_DIMS,GLOBAL_EMBED_LENGTH=256,LOCAL_EMBED_LENGTH=128):
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=EMBED_DIMS,
                      kernel_size=5,
                      stride=1,
                      padding=2), # B,E,2001
            nn.AdaptiveAvgPool1d(GLOBAL_EMBED_LENGTH)
        )
        self.local_projection = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=EMBED_DIMS,
                      kernel_size=5,
                      stride=1,
                      padding=2), # B,E,201
            nn.AdaptiveAvgPool1d(LOCAL_EMBED_LENGTH)
        )
        self.cls_token_global = nn.Parameter(torch.randn(1,1,EMBED_DIMS)) # 1,1,E
        self.cls_token_local = nn.Parameter(torch.randn(1,1,EMBED_DIMS)) # 1,1,E
        
        self.pos_encode_global = nn.Parameter(torch.randn(1,1+GLOBAL_EMBED_LENGTH,EMBED_DIMS)) # 1,L,E
        self.pos_encode_local = nn.Parameter(torch.randn(1,1+LOCAL_EMBED_LENGTH,EMBED_DIMS)) # 1,L,E

    def forward(self,x:torch.Tensor):
        x = x.unsqueeze(1).to(torch.float32)
        BATCH_SIZE = x.shape[0]
        x_global = torch.cat([x[:,:,:2001],x[:,:,2001:4002]],dim=1)
        x_global = self.global_projection(x_global)
        x_global = x_global.transpose(1,2)
        cls_token = self.cls_token_global.expand(BATCH_SIZE,-1,-1)
        x_global = torch.cat([cls_token,x_global],dim=1)
        x_global += self.pos_encode_global

        x_local = torch.cat([x[:,:,4002:4203],x[:,:,4203:4404]],dim=1)
        x_local = self.local_projection(x_local)
        x_local = x_local.transpose(1,2)
        cls_token = self.cls_token_local.expand(BATCH_SIZE,-1,-1)
        x_local = torch.cat([cls_token,x_local],dim=1)
        x_local += self.pos_encode_local

        x_star = x[:,:,4404:]
        x_star = x_star.transpose(1,2)
        
        return x_global,x_local,x_star
    
class MLP(nn.Module):
    '''
    Multilayer Perceptron to use in the transformer encoder block
    Input:
    IN: Input size
    HIDDEN: Hidden size
    DROPOUT: Dropout rate

    Returns a tensor after being passed through MLP
    '''
    def __init__(self,IN,HIDDEN,DROPOUT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(IN,HIDDEN),
            nn.GELU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(HIDDEN,IN),
            nn.Dropout(p=DROPOUT)
        )

    def forward(self,x:torch.Tensor):
        return self.fc(x)

class TransformerEncoder(nn.Module):
    '''
    Transformer encoder block
    Input:
    EMBED_DIMS: Number of embedding dimension
    NUM_HEADS: Number of heads for multihead attention
    DROPOUT: Dropout rate
    HIDDEN: Hidden size of the MLP

    returns a tensor after being passed through a transformer block
    '''
    def __init__(self,EMBED_DIMS,NUM_HEADS,DROPOUT,HIDDEN):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIMS)
        self.attn = nn.MultiheadAttention(EMBED_DIMS,NUM_HEADS,dropout=DROPOUT,batch_first=True)
        self.norm2 = nn.LayerNorm(EMBED_DIMS)
        self.mlp = MLP(EMBED_DIMS,HIDDEN,DROPOUT)

    def forward(self,x:torch.Tensor):
        # LayerNorm is applied before MHA and MLP, this is called pre-layer norm
        x = x + self.attn(self.norm1(x),self.norm1(x),self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    '''
    The full transformer model
    Input:
    EMBED_DIMS: Number of embedding dimension
    NUM_HEADS: Number of heads for multihead attention
    DROPOUT: Dropout rate
    HIDDEN: Hidden size of the MLP
    DEPTH: Number of transformer block
    NUM_CLASSES: Number of classes (1 for Binary)
    '''
    def __init__(self,EMBED_DIMS,NUM_HEADS,DROPOUT,HIDDEN,DEPTH,NUM_CLASSES,GLOBAL_EMBED_LENGTH=256,LOCAL_EMBED_LENGTH=128):
        super().__init__()
        self.embedding = Embedding(EMBED_DIMS,GLOBAL_EMBED_LENGTH,LOCAL_EMBED_LENGTH)
        self.encoder_global = nn.Sequential(
            *[TransformerEncoder(EMBED_DIMS,NUM_HEADS,DROPOUT,HIDDEN) for _ in range(DEPTH)]
        )
        self.encoder_local = nn.Sequential(
            *[TransformerEncoder(EMBED_DIMS,NUM_HEADS,DROPOUT,HIDDEN) for _ in range(DEPTH)]
        )
        self.norm_global = nn.LayerNorm(EMBED_DIMS)
        self.norm_local = nn.LayerNorm(EMBED_DIMS)
        
        # self.head = nn.Linear(EMBED_DIMS,NUM_CLASSES)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIMS*2+6,HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN,HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN,NUM_CLASSES)
        )

    def forward(self,x:torch.Tensor):
        x_global,x_local,x_star = self.embedding(x)
        
        x_global = self.encoder_global(x_global)
        x_global = self.norm_global(x_global) # (B,L,E)

        x_local = self.encoder_local(x_local)
        x_local = self.norm_local(x_local) # (B,L,E)

        token_global = x_global[:,0] # (B,1,E)
        token_local = x_local[:,0] # (B,1,E)

        combined_token = torch.cat([token_global,token_local,x_star.squeeze()],dim=1) # (B,L)
        x = self.head(combined_token)
        return x

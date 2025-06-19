import torch
from torch import nn
from models.layers.grid_attention import *
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels=in_channels

        filter_list=[64,128,256]
        self.down1=DownBlock(self.in_channels,filter_list[0])
        self.down2=DownBlock(filter_list[0],filter_list[1])
        self.down3=DownBlock(filter_list[1],filter_list[2])
        self.down4=DownBlock(filter_list[2],filter_list[-1])
        self.down5=DownBlock(filter_list[-1],filter_list[-1])

    def forward(self,x):
        x,_=self.down1(x)
        x,_=self.down2(x)
        x,_=self.down3(x)
        x,_=self.down4(x)
        x,_=self.down5(x)
        return(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,num_heads,mlp_dimm,embedding_dimm=256,dropout=0.1):
        super().__init__()
        self.embedding_dimm=embedding_dimm
        self.dropout=dropout

        self.norm1=nn.LayerNorm(self.embedding_dimm)
        self.att=nn.MultiheadAttention(self.embedding_dimm,num_heads,dropout=self.dropout,batch_first=True)
        self.norm2=nn.LayerNorm(self.embedding_dimm)

        self.mlp=nn.Sequential(nn.Linear(embedding_dimm,mlp_dimm),
                               nn.GELU(),
                               nn.Dropout(self.dropout),
                               nn.Linear(mlp_dimm,self.embedding_dimm),
                               nn.Dropout(self.dropout))
        
    def forward(self,x):
        """
        Args:
            x: Tensor of shape (B, N, C) where
               B = batch size, N = number of patches/tokens, C = embedding_dim
        Returns:
            x: updated tensor
            attn_weights: attention maps (B, num_heads, N, N)
        """
        x_norm=self.norm1(x)
        attn_out,attn_weights=self.att(x_norm,x_norm,x_norm)
        x=x+attn_out
        x=x+self.mlp(self.norm2(x))
        return x, attn_weights
        
class CNNVit(nn.Module):
    def __init__(self,in_channels,embedding_dim,num_classes=1,num_heads=4):
        super().__init__()
        self.emb_dim=embedding_dim

        filter_list=[128,64,32,16]
        self.encoder=FeatureExtractor(in_channels=in_channels)

        self.att1=TransformerBlock(num_heads,2*self.emb_dim)
        self.att2=TransformerBlock(num_heads,2*self.emb_dim)
        self.att3=TransformerBlock(num_heads,2*self.emb_dim)

        self.up1=nn.ConvTranspose3d(self.emb_dim,filter_list[0],kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0)) # We have learnable upsample of the data, we upsample it to twice the size
        self.up2=nn.ConvTranspose3d(filter_list[0],filter_list[1],kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0))
        self.up3=nn.ConvTranspose3d(filter_list[1],filter_list[2],kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0))
        self.up4=nn.ConvTranspose3d(filter_list[2],filter_list[3],kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0))
        self.up5=nn.ConvTranspose3d(filter_list[3],filter_list[3],kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0))

        self.final=nn.Conv3d(filter_list[3],num_classes,1)

        # This part is a little bit not profesional
        self.pos_embedding=nn.Parameter(torch.rand(1, self.emb_dim, 8, 8, 1))

    def forward(self,x):
        B,C,H,W,D=x.shape

        x=self.encoder(x)

        _, _, H_r, _, _ = x.shape
    
        x=x+self.pos_embedding
        x=x.flatten(2)
        x=x.transpose(1,2)
        x,_=self.att1(x)
        x,_=self.att2(x)
        x,_=self.att3(x)

        # Reshaping tokens into a grid
        x=x.transpose(1,2)
        x=x.reshape(B,x.size()[1],-1,H_r,D)

        # Upsampling part
        x=self.up1(x)
        x=self.up2(x)
        x=self.up3(x)
        x=self.up4(x)
        x=self.up5(x)

        x=self.final(x)
        return x
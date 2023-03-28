import torch, math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BERTConfig:
    hidden_layers: int = 768
    num_heads: int = 12
    attention_blocks: int = 12
    dropout: float = 0.2
    vocabulary_size: int = None
    sequence_len: int = 32

class Head(nn.Module):
    def __init__(self, config: BERTConfig, bias: bool=False):
        """
        Initializes attention head class.

        Args:
            config (BERTConfig): Class with model config data.
            bias (bool, optional): Whether to apply bias to linear layer on no. Defaults to False.
        """
        super().__init__()
        self.W_q = nn.Linear(config.hidden_layers, config.hidden_layers // config.num_heads, bias=bias)
        self.W_k = nn.Linear(config.hidden_layers, config.hidden_layers // config.num_heads, bias=bias)
        self.W_v = nn.Linear(config.hidden_layers, config.hidden_layers // config.num_heads, bias=bias)
        self.dropout = nn.Dropout(p=config.dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of head.

        Args:
            x (torch.Tensor): Input data to be forwarded.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Tensor with applied mask.
        """
        batch_size, t, embeddings = x.shape
        #(batch, t, emb)
        q = self.W_q(x)
        k = self.W_k(x).transpose(1, 2)
        v = self.W_v(x)
        attention = (q @ k) * (1 / math.sqrt(embeddings))
        
        if mask is not None:
            attention = attention.masked_fill((mask.unsqueeze(1) if len(mask.shape) <= 2 else mask) == 0, float('-inf'))
            
        return self.dropout(self.softmax(attention)) @ v

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: BERTConfig):
        """Initializing class computing multiple attention head in pararell.

        Args:
            config (BERTConfig): Model config.
        """
        super().__init__()
        assert config.hidden_layers % config.num_heads == 0, f"Cannot equally distribute {config.hidden_layers} to {config.num_heads} heads!"
        self.attention = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.W_o = nn.Linear(config.hidden_layers, config.hidden_layers, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input data.
            mask (torch.Tensor, optional): Mask to be applied to attention scores. Defaults to None.

        Returns:
            torch.Tensor: Forwarded tensor.
        """
        
        result = torch.cat([head(x, mask) for head in self.attention], dim=-1)
        
        return self.dropout(self.W_o(result))


class Block(nn.Module):
    
    def __init__(self, config: BERTConfig):
        """Initializes transformer block.

        Args:
            config (BERTConfig): Model config.
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.hidden_layers)
        self.norm_2 = nn.LayerNorm(config.hidden_layers)
        self.fully_connected = nn.Sequential(
            nn.Linear(config.hidden_layers, config.hidden_layers*4),
            nn.GELU(),
            nn.Linear(config.hidden_layers*4, config.hidden_layers),
            nn.Dropout(p=config.dropout)
        )
        self.multi_head_att = MultiHeadAttention(config)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """Forward pass of transformer block

        Args:
            x (torch.Tensor): Input data.
            mask (torch.Tensor, optional): Mask to be applied in self attention. Defaults to None.

        Returns:
            torch.Tensor: Forwarded tensor.
        """
        x = x + self.multi_head_att(self.norm_1(x), mask)
        x = x + self.fully_connected(self.norm_2(x))
        return x

class BERTEmbeddings(nn.Module):
    
    def __init__(self, config: BERTConfig):
        """Combines all different forms of embeddings into one class.

        Args:
            config (BERTConfig): Model config.
        """
        super().__init__()
        self.embeddings = nn.Embedding(config.vocabulary_size, config.hidden_layers)
        self.positional_embeddings = nn.Embedding(config.sequence_len, config.hidden_layers)
        self.segment_embeddings = nn.Embedding(2, config.hidden_layers)
        self.emb_norm = nn.LayerNorm(config.hidden_layers)        
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply embeddings to tokenized data.

        Args:
            tokens (torch.Tensor): Tokens to be embedded.
            mask (torch.Tensor): Mask to be used for segment embedding.

        Returns:
            torch.Tensor: Embedded tesnor.
        """
        x = self.embeddings(tokens) + self.positional_embeddings(tokens) + self.segment_embeddings(mask)
        return self.emb_norm(x)

class MLM(nn.Module):
    
    def __init__(self, config: BERTConfig):
        """Initializes Masked Language Modeling (MLM)

        Args:
            config (BERTConfig): Model config.
        """
        super().__init__()
        self.fw = nn.Sequential(
            nn.Linear(config.hidden_layers, config.hidden_layers),
            nn.Tanh(),
            nn.Linear(config.hidden_layers, config.vocabulary_size)
        )
        
    def forward(self, input_data: torch.Tensor, mask: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Calculate loss on masked tokens.

        Args:
            input_data (torch.Tensor): Tensor from BERT model.
            mask (torch.Tensor): Mask applied to the data during forward pass.
            tokens (torch.Tensor): Original tokens.

        Returns:
            torch.Tensor: Calcualted loss on masked tokens.
        """
        logits = self.fw(input_data)
        #extend mask tensor.size([n]) --> tensor.size([n, 1])
        mask = mask.unsqueeze(-1)
        #select only masked logits for loss calculation
        masked_logits = logits.masked_select(mask == 0)
        #select target tokens based on mask
        targets = tokens.unsqueeze(-1).masked_select(mask == 0)
        #calculate loss on masked tokens
        return F.cross_entropy(masked_logits.view(len(targets), -1), targets.view(-1), ignore_index=0)


class BERT(nn.Module):
    
    def __init__(self, config: BERTConfig):
        """BERT model.

        Args:
            config (BERTConfig): Model config.
        """
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.attention_blocks)])
        self.embeddings = BERTEmbeddings(config)
        self.linear = nn.Linear(config.hidden_layers, config.hidden_layers)
        self.mlm = MLM(config)
        self.apply(self._init_weights)
    
    def forward(self,
                tokens: torch.Tensor,
                attention_mask: torch.Tensor=None,
                targets: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the BERT model.

        Args:
            tokens (torch.Tensor): Tokenized data to be embedded.
            attention_mask (torch.Tensor, optional): Mask for attention layers. Defaults to None.
            targets (torch.Tensor, optional): Target classes. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits, loss on MLM task.
        """
        x = self.embeddings(tokens, attention_mask)
        for block in self.blocks:
            x = block(x)
        logits = self.linear(x)
        mlm_loss = None
        if targets is not None:
            mlm_loss = self.mlm.forward(logits, attention_mask, targets)
        return logits, mlm_loss
    
    def training_step(self,
                optimizer: torch.optim.Optimizer, 
                tokens: torch.Tensor,
                attention_mask: torch.Tensor=None,
                targets: torch.Tensor=None) -> int:
        """Function performing forward and backward pass for a batch.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer that is optimizing neural net.
            tokens (torch.Tensor): Tokenized input data.
            attention_mask (torch.Tensor, optional): Mask to be applied to self attention layers. Defaults to None.
            targets (torch.Tensor, optional): Real values for each token. Defaults to None.

        Returns:
            int: Batch loss.
        """
        _, loss = self.forward(tokens, attention_mask, targets)
        self.zero_grad(True)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.15)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.15)

        
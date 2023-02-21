import torch
import torch.nn.functional as f
from dataclasses import dataclass
from itertools import groupby
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt

@dataclass
class NetworkConfig:
    c_embedding: int = 20 # C.shape --> 27 X 20
    vocab_size: int = 27
    batch_size: int = 512
    input_weights: int = 200# Wx.shape --> 20 x 1000; Wx @ x_t --> 512 x 1000
    tanh_gain: float = 5 / 3

@dataclass
class ForwardData:
    X: List
    Y: List
    t: int

class RNN_cell(torch.nn.Module):

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize cell weights and biases tensors.

        Args:
            config (NetworkConfig): Dataclass containing network configuration.
        """
        super().__init__()
        self.Wxt = torch.nn.Linear(config.c_embedding, config.input_weights)
        self.Wat = torch.nn.Linear(config.input_weights, config.input_weights)
        torch.nn.init.kaiming_normal_(self.Wat.weight, nonlinearity='tanh') # samples from N(0, (25 / 9 * config.input_weights)) --> see the docs.
        torch.nn.init.kaiming_normal_(self.Wxt.weight, nonlinearity='tanh') # samples from N(0, (25 / 9 * config.c_embedding)) --> see the docs.

    
    def forward(self, xt: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """Compute forward pass of RNN Cell.

        Args:
            xt (torch.Tensor): Tensor containing input data.
            ht (torch.Tensor): Tensor containing previous context.

        Returns:
            torch.Tensor: Tensor with applied tanh nonlinearity.
        """
        #xt1 = xt (512x20) @ Wxt (20x200) + bx(1,200) -----> (512x200) \
        #                                                               | --> tanh((512,200))
        #h_t = h_t-1 (512x200) @ (200x200) + bh(1,200) ----> (512x200) /
        activation = self.Wxt(xt) + self.Wat(ht)
        return torch.tanh(activation)

class RNN(torch.nn.Module):

    def __init__(self, config: NetworkConfig = NetworkConfig):
        """
        Initialize RNN model with weights and biases.

        Args:
            config (NetworkConfig, optional): Class with model config. Defaults to NetworkConfig.
        """
        super().__init__()
        self.start = torch.nn.Parameter(torch.zeros(1, config.input_weights))
        self.cell = RNN_cell(config)
        self.x_embedding = torch.nn.Embedding(config.vocab_size, config.c_embedding)
        self.W_logits = torch.nn.Linear(config.input_weights, config.vocab_size)        # squash ex: (512x200) @ (200, 27) --> (512, 27)
    
    def forward(self, data: ForwardData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward pass of the model.

        Args:
            data (Tuple[List, List]): Tuple containing list of training samples.
            t (int): Number of timestamps (length of batch sequence).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing logits and loss tensors.
        """
        t = data.t

        batch_size = len(data.X)

        #adjust "dummy" weights for timestamp 0 to fit with batch
        h_prev_context = self.start.expand((batch_size, NetworkConfig.input_weights))

        #create tensor with input data and targets with shape ------------------------------------> (config.batch_size, t)
        xt = torch.tensor(data.X, requires_grad=False).reshape(batch_size, t)      # (config.batch_size, t)
        targets = torch.tensor(data.Y, requires_grad=False).reshape(batch_size, t) # (config.batch_size, t)

        #list containing forward weights for every timestamp
        hidden_states = []                                              # [tensor_1.shape(512, 200), tensor_2.shape(512,200), ...., tensor_t.shape(512, 200)]
        for i in range(t):
            ht = self.cell(self.x_embedding(xt[:, i]), h_prev_context)
            h_prev_context = ht                                         # (batch, 200)
            hidden_states.append(ht)
        
        #Stack all the weights into a tensor and calculate loss
        results = torch.stack(hidden_states, 1) # (config.batch_size, t, config.input_weights)
        logits = self.W_logits(results)         # (config.batch_size, t, config.vocab_size)
        loss = f.cross_entropy(logits.view(-1, NetworkConfig.vocab_size), targets.view(batch_size * t))

        return logits, loss

    def predict(self, start_character: torch.Tensor, func: Callable[[int], str]) -> list[str]:
        """Predicts output based on input character.

        Args:
            start_character (torch.Tensor): Character to start with.
            func (Callable): Function to map int into string.

        Returns:
            list[str]: List with predicted characters.
        """
        self.eval()
        batch_size = 1
        h_prev_context = self.start.expand((batch_size, 200))
        xt = start_character.reshape(1, 1)

        results = [func[start_character.item()]]
        while xt.item():
            ht = self.cell(self.x_embedding(xt.reshape(1,1)), h_prev_context)
            h_prev_context = ht
            logits = self.W_logits(ht)
            probability = torch.softmax(logits.view(1, NetworkConfig.vocab_size), 1)
            result = torch.multinomial(probability, 1, replacement=True)
            results.append(func[result.item()])
            xt = result
        
        self.train(True)

        return results
    
class Dataset:
    def __init__(self, path: str):
        """Initializes object, reads names dataset from path.

        Args:
            path (str): Path to .txt file with data.
        """
        with open(path, "r") as file:
            names = ["." + name.rstrip("\n") + "." for name in filter(lambda name: self._isascii(name.rstrip("\n")), file.readlines())]
        
        self.V = sorted(set(list(''.join(names))))
        self.stoi = {char:num for num, char in enumerate(self.V)}
        self.itos = {num:char for char, num in self.stoi.items()}

        self.data = {length: list(names) for length, names in groupby(sorted(names, key=lambda name: len(name)), len)}

        #drop edge cases for which batch creation would be hard
        for length in [3, 4, *list(range(14, 21))]:
            del self.data[length]
        
        self.lenghts = list(self.data.keys())

        for item in self.lenghts:
            random.shuffle(self.data[item])

    def create_batch(self, t: int) -> ForwardData:
        """Create batch for forward pass of NN.

        Returns:
            ForwardData: Input, output, and sequence length.
        """
        #randomly length of traininig sequence
        # t = torch.randint(min(self.lenghts), max(self.lenghts), (1,)).item()

        #if size of dataset is less than batch use size of dataset
        if (size := len(self.X_train[t])) < NetworkConfig.batch_size:
            batch_s = size
        else:
            batch_s = NetworkConfig.batch_size
        
        #create random numbers
        perm = torch.randperm(size)[:batch_s].clone().detach().tolist()
        
        X_batch, Y_batch = [], []

        for i in perm:
            X_batch.append(self.X_train[t][i])
            Y_batch.append(self.Y_train[t][i])

        return ForwardData(X_batch, Y_batch, t - 1)
    
    def validation(self, t: int, val: bool=True) -> Tuple[List, List]:
        """
        Return validation/test data for sequence length t.

        Args:
            t (int): Length of sequence for which data to be extracted.
            val (bool, optional): Whether to choose from validation or test dataset. Defaults to True.

        Returns:
            Tuple[List, List]: Tuple containig input as index 0 and desired output as index 1.
        """
        return ForwardData(self.X_val[t], self.Y_val[t], t - 1) if val else ForwardData(self.X_test[t], self.Y_test[t], t - 1)
    
    def train_val_test_split(self) -> None:
        """Create train/validation/test data split for training.
        """
        self.X_train, self.Y_train = {k: list() for k in self.lenghts}, {k: list() for k in self.lenghts}
        self.X_val, self.Y_val = {k: list() for k in self.lenghts}, {k: list() for k in self.lenghts} 
        self.X_test, self.Y_test = {k: list() for k in self.lenghts}, {k: list() for k in self.lenghts} 

        for item in self.lenghts:
            words_list_len = len(self.data[item])
            train = int(words_list_len * 0.8)
            val = int(words_list_len * 0.9)
            for name in self.data[item][:train]:
                self.X_train[item].append([self.stoi[char] for char in name[:-1]])
                self.Y_train[item].append([self.stoi[char] for char in name[1:]])
            for name in self.data[item][train:val]:
                self.X_val[item].append([self.stoi[char] for char in name[:-1]])
                self.Y_val[item].append([self.stoi[char] for char in name[1:]])
            for name in self.data[item][val:]:
                self.X_test[item].append([self.stoi[char] for char in name[:-1]])
                self.Y_test[item].append([self.stoi[char] for char in name[1:]])
        
        assert self._dict_list_len(self.data) == (self._dict_list_len(self.X_test) + self._dict_list_len(self.X_train) + self._dict_list_len(self.X_val))
    
    def __len__(self) -> int:
        """Returns number of words in dataset.

        Returns:
            int: Number of names in dataset.
        """
        return self._dict_list_len(self.data)
    
    def __repr__(self) -> str:
        return f"V --> {self.V}\n|V| --> {len(self.V)}\nNumber of names in dataset -->{len(self)}\nNumber of\n\ttrain examples --> {self._dict_list_len(self.X_train)}\n\tvalidation examples --> {self._dict_list_len(self.X_val)}\n\ttest examples --> {self._dict_list_len(self.X_test)}"
    
    
    def get_sequences_length(self) -> list[int]:
        """Returns lengths of sequences in data.

        Returns:
            list[int]: List with sequences length.
        """
        return self.lenghts

    @staticmethod
    def _dict_list_len(dictionary: dict) -> int:
        """Counts number of items in each dictionary value.

        Args:
            dictionary (dict): Dict containing data.

        Returns:
            int: Summed up number of items in dataset.
        """
        counter = 0
        for k in dictionary:
            counter += len(dictionary[k])
        return counter

    @staticmethod
    def _isascii(string: str) -> bool:
        for ch in string:
            if ord(ch) not in range(97, 123): return False
        return True

class RDataset(Dataset):

    def __init__(self, data: List[str], characters: List[str], max_len: int) -> None:
        self.words = data
        self.characters = characters
        self.max_word_len = max_len
        self.stoi = {char:val + 1 for val, char in enumerate(self.characters)}
        self.itos = {val:char for char, val in self.stoi.items()}
    
    def __len__(self):
        return len(self.words)
    
    def encode(self, word: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in word], dtype=torch.long, requires_grad=False)
    
    def __getitem__(self, index):
        word = self.words[index]
        index = self.encode(word)
        #pad the examples
        x = torch.zeros(self.max_word_len + 1, dtype=torch.long, requires_grad=False)
        y = torch.zeros_like(x, requires_grad=False)
        x[1:len(index) + 1] = index
        y[:len(index)] = index
        y[len(index) + 1:] = -1
        return x, y


def _isascii(string: str) -> bool:
    for ch in string:
        if ord(ch) not in range(97, 123): return False
    return True

def plot_learning(title: str, x_label: str, y_label: str, epochs: List[int], **kwargs) -> None:
    plt.figure(figsize=(11,7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(epochs, kwargs['train_loss'], label="Training loss")
    plt.plot(epochs, kwargs['validation_loss'], label="Validation loss")
    plt.legend()
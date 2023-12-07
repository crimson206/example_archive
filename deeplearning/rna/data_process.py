import pandas as pd
import torch
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

def tokenize(series: pd.Series, max_len: int) -> torch.Tensor:
    '''
    Tokenizes a pandas Series of nucleotide sequences into a tensor, with each nucleotide converted to a unique integer.
    Sequences shorter than `max_len` are padded with zeros.

    Parameters:
    series (pd.Series): A series of nucleotide sequences.
    max_len (int): The desired length for the sequences after padding.

    Returns:
    torch.Tensor: A tensor containing the padded, tokenized sequences.
    '''
    # Extract all unique letters from the series to create a tokenizer dictionary
    unique_letters = sorted(set(series.str.cat()))
    tokenizer = {letter: idx+1 for idx, letter in enumerate(unique_letters)}

    pad_val = len(unique_letters) + 1

    # Initialize a list to store tokenized sequences
    tokenized_sequences = []

    # Tokenize each sequence in the series
    for sequence in tqdm(series, total=len(series), desc="Tokenizing sequences"):
        # Tokenize the sequence
        tokenized_seq = [tokenizer[letter] for letter in sequence]
        # Pad the sequence with 0s if it's shorter than seq_len
        padded_seq = tokenized_seq + [pad_val] * (max_len - len(tokenized_seq))
        # Ensure the sequence is cut off at the seq_len
        tokenized_sequences.append(padded_seq[:max_len])

    # Convert the list of tokenized sequences to a torch tensor
    return torch.tensor(tokenized_sequences)

def create_attention_mask(data, max_len):

    # Check if there is only one unique sequence length
    unique_seq_lens = data["seq_len"].unique()
    if len(unique_seq_lens) != 1:
        raise ValueError(f"Expected one unique sequence length, but found {len(unique_seq_lens)}")

    seq_len = unique_seq_lens[0]
    n_data = len(data)

    attention_mask = torch.zeros((n_data, max_len))
    attention_mask[:,:seq_len] = 1

    return attention_mask

def create_continuous_feature(data: pd.DataFrame, target_col_str: str, max_len: int) -> torch.Tensor:

    data = data.copy()
    unique_seq_lens = data["seq_len"].unique()
    if len(unique_seq_lens) != 1:
        raise ValueError(f"Expected one unique sequence length, but found {len(unique_seq_lens)}")

    # Select columns that match the target column string
    target_cols = [col for col in data.columns if target_col_str in col]

    processed_features = data[target_cols].values[:,:max_len] 

    # Convert the list of features to a torch tensor
    return torch.tensor(processed_features, dtype=torch.float)

def create_nan_label(label, seq_len):
    nan_label = torch.isnan(label).float()
    nan_label[:,seq_len:] = torch.nan
    return nan_label.long()

def create_position_ids(data: pd.DataFrame, max_len: int, direction: str = "front") -> torch.Tensor:
    """
    Checks for a unique sequence length in the provided DataFrame and creates a tensor 
    of position IDs for each sequence in the DataFrame, either from the front or the back. 
    If 'front', each position ID starts from 1 up to the sequence length and pads the rest with 0.
    If 'back', the positions are reversed starting from the sequence length down to 1, 
    and pads the rest with 0.

    Parameters:
    data (pd.DataFrame): A DataFrame containing a 'seq_len' column with sequence lengths.
    max_len (int): The maximum length for the sequences after padding.
    direction (str): The direction to create position IDs, either 'front' or 'back'.

    Raises:
    ValueError: If there is not exactly one unique sequence length in 'seq_len' column.

    Returns:
    torch.Tensor: A tensor of position IDs for each sequence.
    """
    # Check if there is only one unique sequence length
    unique_seq_lens = data["seq_len"].unique()
    if len(unique_seq_lens) != 1:
        raise ValueError(f"Expected one unique sequence length, but found {len(unique_seq_lens)}")
    
    # Get the unique sequence length
    seq_len = unique_seq_lens[0]

    # Initialize the tensor to hold the position IDs for all sequences
    if direction == "front":
        # Create a range for the actual sequence positions starting from 1
        position_ids = torch.arange(1, seq_len+1).unsqueeze(0).repeat(len(data), 1)
    elif direction == "back":
        # Create a range for the actual sequence positions starting from seq_len
        position_ids = torch.arange(seq_len, 0, -1).unsqueeze(0).repeat(len(data), 1)
    else:
        raise ValueError(f"Invalid direction '{direction}'. Expected 'front' or 'back'.")

    # Pad the sequence with 0 if it's shorter than max_len
    pad_val = position_ids.max() + 1
    pad = pad_val * torch.ones(len(data), max_len - seq_len, dtype=torch.long)
    position_ids = torch.cat((position_ids, pad), dim=1)
    
    return position_ids


# Create adjacency matrix based on structure string
def structure_to_pair_matrix(struct):
    n = len(struct)
    adj_matrix = torch.zeros((n, n))
    stack = []

    # Add connections based on brackets
    for i, char in enumerate(struct):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    return adj_matrix


def create_pair_matrix(structures, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in structures)  # Find the length of the longest structure
    padded_adj_matrices = []

    for struct in structures:
        adj_matrix = structure_to_pair_matrix(struct)
        # Pad the adjacency matrix to have the same size as max_len
        pad_size = max_len - len(struct)
        padded_adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size), "constant", -1000)
        padded_adj_matrices.append(padded_adj_matrix)

    return torch.stack(padded_adj_matrices).long()

def tokenize_batch(series, seq_len):
    tokenizer, _ = pd.factorize(series)
    return torch.tensor(tokenizer).unsqueeze(-1).expand(-1, seq_len)

def generate_column(data, column_str, seq_len, log=False):
    if log:
        noise_index = torch.tensor(np.log(data[column_str].values).astype(float))
    else:
        noise_index = torch.tensor(data[column_str].values.astype(float))
    return noise_index.float()#.unsqueeze(-1).expand(-1, seq_len).float()

def generate_reads_log(data, seq_len):
    reads_log = (np.log(data["reads"].values) / np.log(1.2)).astype(int)
    return torch.tensor(reads_log).long().unsqueeze(-1).expand(-1, seq_len)

def process_data(df, seq_len, max_len=None):
    if max_len is None:
        max_len = seq_len

    print("processing seq_len", seq_len)

    tensor_dict = {}
    data = df[df["seq_len"]==seq_len]
    tensor_dict["sequence_ids"] = tokenize(data["sequence"], max_len=max_len)
    tensor_dict["structure_ids"] = tokenize(data["structure"], max_len=max_len)
    tensor_dict["experiment_ids"] = tokenize_batch(data["experiment_type"], seq_len)
    tensor_dict["attention_mask"] = create_attention_mask(data, max_len=max_len)
    tensor_dict["reactivity"] = create_continuous_feature(data, "reactivity_0", max_len=max_len)
    tensor_dict["error"] = create_continuous_feature(data, "reactivity_error", max_len=max_len)
    tensor_dict["nan_position"] = create_nan_label(tensor_dict["reactivity"], seq_len)
    tensor_dict["front_position_ids"] = create_position_ids(data, max_len)
    tensor_dict["back_position_ids"] = create_position_ids(data, max_len, direction="back")
    tensor_dict["reads_log"] = generate_column(data, "reads", seq_len, log=True)
    tensor_dict["signal_to_noise_log"] = generate_column(data, "signal_to_noise", seq_len, log=True)

    return tensor_dict

def dict_to_device(data, device, dtype=torch.float16):
    new_data = {}
    for key, tensor in data.items():
        tensor = tensor.to(device)
        if dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        new_data[key] = tensor
    return new_data

def create_batch(tensor_dict):

    seq_len = tensor_dict["structure_ids"].shape[1]

    split_position_ids = [
        tensor_dict["front_position_ids"],
        tensor_dict["back_position_ids"],
    ]

    split_categorical_ids = [
        tensor_dict["sequence_ids"],
        tensor_dict["structure_ids"],
        tensor_dict["experiment_ids"],
    ]

    skip_embedding = torch.stack(
        [
            tensor_dict["reads_log"][:,0].float(),
            tensor_dict["signal_to_noise"][:,0],
        ],
        dim=-1
    )

    labels = [
        tensor_dict["nan_position"],
        tensor_dict["error"],
        tensor_dict["reactivity"],
        torch.zeros(tensor_dict["error"].shape),
    ]

    batch1 = {
        "position":split_position_ids,
        "category":split_categorical_ids,
        "skip_emb":skip_embedding,
        "label":labels
    }

    return batch1
import torch
import torch.nn.functional as F
from typing import List, Tuple, Literal, Optional


def concatenate_and_pad(
    list_of_tensors: List[torch.Tensor],
    pad_val: float,
    padding_side: str = 'left'
) -> Tuple[torch.Tensor, List[int]]:
    """
    Concatenates a list of 1D tensors into a 2D tensor, padding them
    to the length of the longest tensor in the list.

    Args:
        list_of_tensors (List[torch.Tensor]): A list of 1D tensors of varying lengths.
        pad_val (float): The value to use for padding.
        padding_side (str, optional): The side to pad on. Must be 'left' or 'right'.
                                      Defaults to 'left'.

    Returns:
        Tuple[torch.Tensor, List[int]]: A tuple containing:
            - The resulting 2D tensor of shape (num_tensors, max_length).
            - A list of integers representing the padding amount applied to each tensor.
    """
    if padding_side not in ['left', 'right']:
        raise ValueError("padding_side must be either 'left' or 'right'.")
    if not list_of_tensors:
        return torch.empty(0, 0), []

    max_len = max(t.size(0) for t in list_of_tensors)
    padded_tensors = []
    padding_amounts = []

    for t in list_of_tensors:
        pad_amount = max_len - t.size(0)
        padding_amounts.append(pad_amount)

        pad_tuple = (pad_amount, 0) if padding_side == 'left' else (0, pad_amount)
        
        padded_t = F.pad(t, pad_tuple, mode='constant', value=pad_val)
        padded_tensors.append(padded_t)

    concatenated_tensor = torch.stack(padded_tensors)
    return concatenated_tensor, padding_amounts


def apply_padding_to_list(
    tensors: List[torch.Tensor],
    padding_amounts: List[int],
    pad_val: float,
    padding_side: str = 'left'
) -> List[torch.Tensor]:
    """
    Applies padding to a new list of tensors based on a pre-calculated
    list of padding amounts.

    Args:
        second_list_of_tensors (List[torch.Tensor]): The new list of 1D tensors to pad.
        padding_amounts (List[int]): A list of padding amounts.
        pad_val (float): The value to use for padding.
        padding_side (str, optional): The side to pad on. Must be 'left' or 'right'.
                                      Defaults to 'left'.

    Returns:
        List[torch.Tensor]: A new list of padded 1D tensors.
    """
    if padding_side not in ['left', 'right']:
        raise ValueError("padding_side must be either 'left' or 'right'.")
    if len(tensors) != len(padding_amounts):
        raise ValueError("The number of tensors must match the number of padding amounts.")

    result_list = []
    for tensor, pad_amount in zip(tensors, padding_amounts):
        if pad_amount > 0:
            padding_tensor = torch.full(
                (pad_amount,),
                pad_val,
                dtype=tensor.dtype,
                device=tensor.device
            )
            # Adjust concatenation order based on padding_side
            if padding_side == 'left':
                new_tensor = torch.cat([padding_tensor, tensor], dim=0)
            else: # padding_side == 'right'
                new_tensor = torch.cat([tensor, padding_tensor], dim=0)
            result_list.append(new_tensor)
        else:
            result_list.append(tensor)

    return result_list


def apply_padding(
    sequences: List[torch.Tensor],
    pad_value: int,
    direction: Literal["left", "right"] = "left",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of 1D tensors and returns the padded tensor and its attention mask.

    This function takes a list of tensors, finds the maximum length, and pads
    each tensor on either the left or the right to match that length.

    Args:
        sequences: A list of 1D tensors.
        pad_value: The value to use for padding.
        direction: The padding direction, either "left" or "right".
            Defaults to "left".

    Returns:
        A tuple containing:
        - A 2D tensor of shape (num_sequences, max_length) with the padded sequences.
        - A 2D attention mask tensor of the same shape, with 1s for original tokens
          and 0s for padding tokens.
    """
    if direction not in ["left", "right"]:
        raise ValueError("Direction must be either 'left' or 'right'.")

    if not sequences:
        # Return empty tensors if the input list is empty
        return torch.empty((0, 0), dtype=torch.long), torch.empty(
            (0, 0), dtype=torch.long
        )

    device = sequences[0].device
    max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    attention_masks = []

    for seq in sequences:
        num_pads = max_len - len(seq)

        if num_pads > 0:
            # Create the padding tensor for the sequence
            padding_tensor = torch.full(
                (num_pads,), pad_value, device=device, dtype=torch.long
            )
            # Create the corresponding mask for the padding
            padding_mask = torch.zeros(num_pads, device=device, dtype=torch.long)
            # Create the mask for the original sequence content
            content_mask = torch.ones(len(seq), device=device, dtype=torch.long)

            if direction == "left":
                padded_seq = torch.cat([padding_tensor, seq])
                mask = torch.cat([padding_mask, content_mask])
            else:  # direction == "right"
                padded_seq = torch.cat([seq, padding_tensor])
                mask = torch.cat([content_mask, padding_mask])
        else:
            # No padding needed if the sequence is already at max length
            padded_seq = seq
            mask = torch.ones(len(seq), device=device, dtype=torch.long)

        padded_sequences.append(padded_seq)
        attention_masks.append(mask)

    return torch.stack(padded_sequences, dim=0), torch.stack(attention_masks, dim=0)


def remove_special_tokens(
    batch_tensor: torch.Tensor,
    pad_token: Optional[int] = None,
    bos_token: Optional[int] = None,
    eos_token: Optional[int] = None,
    keep_trailing_eos: bool = False,
) -> List[torch.Tensor]:
    """
    Remove padding, BOS, and EOS tokens from a batch of token sequences.

    This function iterates through a batch of sequences and removes specified
    special tokens. It includes an option to preserve a single, trailing
    EOS token if one exists at the end of the non-padding sequence.

    Args:
        batch_tensor: A 2D tensor of shape (batch_size, seq_len).
        pad_token: The padding token to remove. Defaults to None.
        bos_token: The beginning-of-sequence token to remove. Defaults to None.
        eos_token: The end-of-sequence token to remove. Defaults to None.
        keep_trailing_eos: If True, preserves a single trailing EOS token. Defaults to False.

    Returns:
        A list of 1D tensors, where each tensor is a sequence with special
        tokens removed according to the specified rules.
    """
    cleaned_sequences = []
    for seq in batch_tensor:
        # Find the index of the specific trailing EOS to preserve
        eos_to_keep_index = -1
        if keep_trailing_eos and eos_token is not None:
            # Iterate backwards from the end of the sequence.
            for i in range(len(seq) - 1, -1, -1):
                # Check if the current token is a "content" token.
                is_eos = seq[i].item() == eos_token
                is_pad = pad_token is not None and seq[i].item() == pad_token

                if not is_eos and not is_pad:
                    # This is the last content token, at index `i`.
                    # Now, check if the *next* token is the EOS we want to keep.
                    if (i + 1) < len(seq) and seq[i + 1].item() == eos_token:
                        eos_to_keep_index = i + 1
                    # We've found our anchor point, so we can stop searching.
                    break

        # Start with a mask that keeps all tokens.
        keep_mask = torch.ones_like(seq, dtype=torch.bool)

        # Mark special tokens for removal.
        if pad_token is not None:
            keep_mask[seq == pad_token] = False
        if bos_token is not None:
            keep_mask[seq == bos_token] = False
        if eos_token is not None:
            keep_mask[seq == eos_token] = False

        # If we identified a trailing EOS to keep, override the mask at that specific index.
        if eos_to_keep_index != -1:
            keep_mask[eos_to_keep_index] = True

        # Apply the final mask to the sequence.
        cleaned_sequences.append(seq[keep_mask])

    return cleaned_sequences


def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Creates position IDs from an attention mask.

    Position IDs are a cumulative sum over the attention mask, which correctly
    increments position for non-padding tokens and keeps it the same for padding.

    Args:
        attention_mask: A 2D tensor of shape (batch_size, seq_len) with 0s for
            padding and 1s for content tokens.

    Returns:
        A 2D tensor of the same shape containing the calculated position IDs.
    """
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
    return position_ids.long()


def detect_subsequence(
    batch_tensor: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Detects if the target sequence is present in each row of x.

    Args:
        batch_tensor: 2-D tensor of shape (batch_size, seq_len)
        target: 1-D tensor of length target_len

    Returns:
        A 1-D tensor of shape (batch_size,) with 1s where the target is found in the row, 0s otherwise.
    """
    bz, seq_len = batch_tensor.shape
    target_len = target.size(0)

    if target_len > seq_len:
        return torch.zeros(bz, dtype=torch.long, device=batch_tensor.device)

    # Create sliding windows of shape (bz, seq_len - target_len + 1, target_len)
    windows = batch_tensor.unfold(
        1, target_len, 1
    )  # shape: (bz, num_windows, target_len)

    # Compare each window to the target
    matches = (windows == target.unsqueeze(0).unsqueeze(0)).all(dim=2)

    # Check if any window in each row matches the target
    result = matches.any(dim=1).long()

    return result


def split_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a mask into two masks:
    - mask_1: 1s before the first 0 in each row (0s elsewhere). If no 0s, all 1s.
    - mask_2: 1s after the first 0 in each row (0s elsewhere, or all 0s if no 1s after 0 or no 0s).
    This is used for splitting generator response masks into the initial and completion parts.

    Args:
        mask: A 2D tensor of shape (batch_size, seq_len) with 0s for padding and 1s for content tokens.

    Returns:
        A tuple containing:
        - A new mask of shape (batch_size, seq_len) with 1s before the first 0 in each row (0s elsewhere).
        If no 0s, all 1s.
        - A new mask of shape (batch_size, seq_len) with 1s after the first 0 in each row
        (0s elsewhere, or all 0s if no 1s after 0 or no 0s).
    """
    bz, seq_len = mask.shape

    # Find the index of the first 0 in each row (or seq_len if no 0 exists)
    zero_mask = mask == 0
    has_zero = zero_mask.any(dim=1)  # Shape: (bz,)
    first_zero = torch.where(
        has_zero,
        zero_mask.int().argmax(dim=1),
        torch.tensor(seq_len, device=mask.device),
    )

    # Create mask_1 (1s before first 0, 0s elsewhere)
    arange = torch.arange(seq_len, device=mask.device).expand(bz, -1)  # (bz, seq_len)
    mask_1 = (arange < first_zero.unsqueeze(1)).float()  # 1s before first 0

    # Create mask_2 (1s after first 0, but only if there are 1s after the first 0)
    last_one = seq_len - 1 - (mask.flip(dims=[1]) == 1).int().argmax(dim=1)  # (bz,)
    has_ones_after_zero = (
        last_one > first_zero
    ) & has_zero  # True if 1s after first 0 AND row has at least one 0
    # mask_2 is 1s where (position >= first_zero) AND (row has 1s after first 0) AND (original mask is 1)
    mask_2 = (
        (arange >= first_zero.unsqueeze(1))
        & has_ones_after_zero.unsqueeze(1)
        & (mask == 1)
    ).float()

    return mask_1, mask_2


def trim_padding(tensor: torch.Tensor, pad_value: int, direction: str = "right") -> torch.Tensor:
    """
    Trims excess padding from a 2D tensor, handling cases where padding may exist on both sides.

    Args:
        tensor (torch.Tensor): The 2D input tensor (B, L).
        pad_value (int): The integer value representing padding.
        direction (str): The primary direction of padding to trim ("left" or "right").

    Returns:
        torch.Tensor: The trimmed and re-padded tensor.
    """
    assert tensor.dim() == 2, "Input must be 2D"
    assert direction in {"left", "right"}, "direction must be 'left' or 'right'"

    B, L = tensor.shape
    device = tensor.device
    non_pad_mask = tensor != pad_value

    if direction == "left":
        first_non_pad = torch.where(non_pad_mask, torch.arange(L, device=device), L).min(dim=1).values
        lengths = L - first_non_pad
        max_len = lengths.max().item()

        rows = []
        for i in range(B):
            seq = tensor[i, first_non_pad[i]:]
            pad_needed = max_len - seq.size(0)
            if pad_needed > 0:
                pad_tensor = torch.full((pad_needed,), pad_value, device=device)
                seq = torch.cat([pad_tensor, seq])
            rows.append(seq)

    else:
        indices = torch.arange(L, device=device).expand_as(tensor)
        last_non_pad_indices = torch.where(non_pad_mask, indices, -1).max(dim=1).values
        
        new_lengths = torch.clamp(last_non_pad_indices + 2, min=1, max=L)
        max_len = new_lengths.max().item()

        rows = []
        for i in range(B):
            length = new_lengths[i].item()
            seq = tensor[i, :length]
            pad_needed = max_len - length
            if pad_needed > 0:
                pad_tensor = torch.full((pad_needed,), pad_value, device=device)
                seq = torch.cat([seq, pad_tensor])
            rows.append(seq)

    return torch.stack(rows, dim=0)


def trim_right_pad_column(x: torch.Tensor, pad_val):
    """
    x: (n, d) tensor
    pad_val: scalar pad value
    
    Returns:
        trimmed_x: (n, d') tensor
        num_dropped: number of columns removed from the right
    """
    n, d = x.shape

    # True for columns that have any non-pad value
    nonpad_any = (x != pad_val).any(dim=0)  # (d,)

    # If all columns contain some non-pad value → drop 0 columns
    if nonpad_any.all():
        return x, 0

    # First column that is entirely pad
    first_pad_idx = torch.nonzero(~nonpad_any, as_tuple=False)[0].item()

    # We want to keep exactly ONE pad column at the right
    keep = first_pad_idx + 1
    num_dropped = d - keep

    trimmed_x = x[:, :keep]
    return trimmed_x, num_dropped


def drop_last_columns(x: torch.Tensor, num_cols: int):
    """
    Removes the last `num_cols` columns of x.
    If num_cols = 0, returns x unchanged.
    """
    if num_cols <= 0:
        return x
    return x[:, :-num_cols]
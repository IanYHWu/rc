import os
import pickle
import torch
import numpy as np

def convert_data_proto(pkl_path, output_dir):
    # Load the DataProto object (requires verl to be installed)
    print(f"Loading DataProto from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data_proto = pickle.load(f)

    # Access batches
    batch = data_proto.batch  # should be a TensorDict
    non_tensor_batch = data_proto.non_tensor_batch  # should be plain Python data (list, dict, etc.)

    # Convert TensorDict -> plain dict of torch tensors
    print("Converting TensorDict to plain dict...")
    tensor_dict = {}
    for key, value in batch.items():
        # Ensure tensors are detached and moved to CPU
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value.detach().cpu()
        else:
            # If something weird sneaks in, convert to tensor if possible
            try:
                tensor_dict[key] = torch.as_tensor(value)
            except Exception:
                tensor_dict[key] = value

    # Convert non-tensor batch entries to numpy if possible
    print("Converting non-tensor data...")
    numpy_dict = {}
    if non_tensor_batch is not None:
        for key, value in non_tensor_batch.items():
            if isinstance(value, torch.Tensor):
                numpy_dict[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                numpy_dict[key] = value
            else:
                try:
                    numpy_dict[key] = np.array(value)
                except Exception:
                    numpy_dict[key] = value  # leave as-is if it can’t be converted

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save tensors as a .pt file (for easy PyTorch loading)
    torch_path = os.path.join(output_dir, "batch.pt")
    torch.save(tensor_dict, torch_path)

    # Save non-tensors as a .npz file (for easy NumPy loading)
    npz_path = os.path.join(output_dir, "non_tensor_batch.npz")
    np.savez_compressed(npz_path, **numpy_dict)

    print(f"Saved torch tensors to: {torch_path}")
    print(f"Saved numpy arrays to: {npz_path}")
    print("✅ Conversion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DataProto to portable tensors and numpy arrays.")
    parser.add_argument("--input", type=str, required=True, help="Path to the .pkl DataProto file.")
    parser.add_argument("--output", type=str, default="converted_data", help="Directory to save output files.")
    args = parser.parse_args()

    convert_data_proto(args.input, args.output)

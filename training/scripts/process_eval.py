

import pandas as pd
import numpy as np
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Path to the JSON file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the JSON file as a pandas DataFrame
    df = pd.read_json(args.filename)

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # tokenizer.model_max_length = 32768
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    print(f"Dataset size: {len(df)}")
    print("Applying tokenization...")
    # Define a function to calculate the length of the prompt and completion
    def get_encoded_lengths(record):
        return len(tokenizer.encode(record['output']))
    
    # Apply the function to the DataFrame
    df['len'] = df.apply(get_encoded_lengths, axis=1)

    df['under_2k'] = ((df['len'] < 2048) * (df['score'])).to_numpy(dtype=float)
    df['under_2k_pass@8'] = df.groupby("index")["under_2k"].transform("max")
    
    df['under_4k'] = ((df['len'] < 4096) * (df['score'])).to_numpy(dtype=float)
    df['under_4k_pass@8'] = df.groupby("index")["under_4k"].transform("max")
    
    df['under_8k'] = ((df['len'] < 8192) * (df['score'])).to_numpy(dtype=float)
    df['under_8k_pass@8'] = df.groupby("index")["under_8k"].transform("max")

    df['under_16k'] = ((df['len'] < 16384) * (df['score'])).to_numpy(dtype=float)
    df['under_16k_pass@8'] = df.groupby("index")["under_16k"].transform("max")

    df['under_24k'] = ((df['len'] < 24576) * (df['score'])).to_numpy(dtype=float)
    df['under_24k_pass@8'] = df.groupby("index")["under_24k"].transform("max")

    df['under_32k'] = ((df['len'] < 32768) * (df['score'])).to_numpy(dtype=float)
    df['under_32k_pass@8'] = df.groupby("index")["under_32k"].transform("max")


    # Print the DataFrame
    print(df.head())
    print("2k:", df['under_2k'].mean(), "pass@8:", df['under_2k_pass@8'].mean())
    print("4k:", df['under_4k'].mean(), "pass@8:", df['under_4k_pass@8'].mean())
    print("8k:", df['under_8k'].mean(), "pass@8:", df['under_8k_pass@8'].mean())
    print("16k:", df['under_16k'].mean(), "pass@8:", df['under_16k_pass@8'].mean())
    print("24k:", df['under_24k'].mean(), "pass@8:", df['under_24k_pass@8'].mean())
    print("32k:", df['under_32k'].mean(), "pass@8:", df['under_32k_pass@8'].mean())

    
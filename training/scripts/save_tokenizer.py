from transformers import AutoTokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_path', type=str, required=True, help="The path for the huggingface model")
    parser.add_argument('--hf_upload_path', type=str, required=True, help="The path to save the tokenizer")
    args = parser.parse_args()
    
    print(f"Saving tokenizer to {args.hf_upload_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.push_to_hub(args.hf_upload_path)

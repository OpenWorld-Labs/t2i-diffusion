#!/usr/bin/env python3
from datasets import load_dataset
from pathlib import Path
import os

def main():
    # Set the output directory
    output_dir = Path("data/imagenet-1k")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ImageNet-1K from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("imagenet-1k")
    
    # Save the dataset
    print("Saving dataset...")
    dataset.save_to_disk(str(output_dir))
    
    print(f"\nDataset downloaded and saved to {output_dir}")
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")

if __name__ == "__main__":
    main() 
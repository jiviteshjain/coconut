#!/usr/bin/env python3
"""
Simple inference script for Coconut model.
Usage: python inference.py --checkpoint_path /path/to/checkpoint --model_id openai-community/gpt2
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut


def load_model_and_tokenizer(checkpoint_path, model_id, device="cuda"):
    """Load the Coconut model and tokenizer from checkpoint."""
    
    # Load the base model and tokenizer
    print(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Add special tokens for Coconut (must be in same order as training)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens in the exact same order as training script
    # This ensures token IDs match the training run
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>") 
    tokenizer.add_tokens("<|latent|>")
    
    # Get token IDs (should match training run)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    print(f"Special token IDs - latent: {latent_id}, start: {start_id}, end: {end_id}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Resize embeddings to accommodate new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Wrap with Coconut
    print("Wrapping model with Coconut...")
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Load Coconut checkpoint
    print(f"Loading Coconut checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def format_question_with_latent_tokens(question, tokenizer, c_thought=2):
    """Format a question with latent tokens for Coconut inference."""
    
    # Format exactly like the training script: question + start + latent*k + end
    latent_tokens = "<|latent|>" * c_thought
    formatted_question = f"{question}<|start-latent|>{latent_tokens}<|end-latent|>"
    
    return formatted_question

def logit_lens(model, tokenizer, outputs, embeddings):
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    indices_with_latents = (outputs[0] == latent_id).nonzero()[:, 0]  # shape (num_latents, )
    latent_embeddings = embeddings[0, indices_with_latents, :]  # shape (num_latents, embedding_dim)
    logits = model.base_causallm.lm_head(latent_embeddings)  # shape (num_latents, vocab_size)
    _, top_k_token_indices = torch.topk(logits, k=5, dim=1)  # shape (num_latents, 10)
    print(f"Top 5 tokens for each latent:")
    for i in range(top_k_token_indices.shape[0]):
        tokens = tokenizer.convert_ids_to_tokens(top_k_token_indices[i])
        print(f"Latent {i}: {tokens}")


def generate_response(model, tokenizer, question, max_new_tokens=128, c_thought=2, device="cuda"):
    """Generate a response using the Coconut model."""
    
    # Format the question with latent tokens
    formatted_question = format_question_with_latent_tokens(question, tokenizer, c_thought)
    
    # Tokenize
    inputs = tokenizer(formatted_question, return_tensors="pt").to(device)
    
    print(f"Input question: {formatted_question}")
    print(f"Input tokens: {inputs['input_ids'][0].tolist()}")
    
    # Generate
    with torch.no_grad():
        outputs, embeddings = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            synced_gpus=False,  # Set to False for single GPU inference
            output_embedding=True
        )
    
    logit_lens(model, tokenizer, outputs, embeddings)
    
    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the input)
    input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    generated_text = full_response[input_length:].strip()
    
    return full_response, generated_text


def main():
    parser = argparse.ArgumentParser(description="Coconut Model Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to the model checkpoint")
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2",
                       help="Base model ID (default: openai-community/gpt2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--c_thought", type=int, default=2,
                       help="Number of latent thought tokens to use")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint_path, 
        args.model_id, 
        args.device
    )
    
    print("\n" + "="*50)
    print("Coconut Model Inference Ready!")
    print("="*50)
    print(f"Model: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Latent thoughts: {args.c_thought}")
    print("="*50)
    
    # Interactive inference loop
    while True:
        try:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not question:
                continue
                
            print("\nGenerating response...")
            full_response, generated_text = generate_response(
                model, tokenizer, question, 
                max_new_tokens=args.max_new_tokens,
                c_thought=args.c_thought,
                device=args.device
            )
            
            print(f"\nFull response:\n{full_response}")
            print(f"\nGenerated text:\n{generated_text}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()

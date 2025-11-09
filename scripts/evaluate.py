"""
Evaluation script for trained Transformer model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from typing import List
from src.models.transformer import Transformer
from src.data.dataset import Vocabulary, tokenize, load_multi30k_data
from src.config import Config

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")


def compute_bleu(predictions: List[str], references: List[str]) -> dict:
    """
    Compute BLEU score using sacrebleu.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
    
    Returns:
        Dictionary with BLEU scores
    """
    if not SACREBLEU_AVAILABLE:
        return {"bleu": 0.0, "error": "sacrebleu not available"}
    
    # sacrebleu.corpus_bleu expects: predictions (list of str), references (list of list of str)
    # For single reference: [[ref1], [ref2], [ref3], ...]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    return {
        "bleu": bleu.score,
        "bleu_1": bleu.precisions[0],
        "bleu_2": bleu.precisions[1],
        "bleu_3": bleu.precisions[2],
        "bleu_4": bleu.precisions[3],
    }


def translate(
    model,
    src_text: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 128
) -> str:
    """
    Translate a source sentence to target language.
    
    Args:
        model: Trained Transformer model
        src_text: Source sentence
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device
        max_len: Maximum generation length
    
    Returns:
        Translated sentence
    """
    model.eval()
    
    # Tokenize and encode source
    src_tokens = tokenize(src_text, 'en')
    src_indices = [src_vocab.bos_idx] + src_vocab.encode(src_tokens) + [src_vocab.eos_idx]
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    
    # Generate translation
    with torch.no_grad():
        output_indices = model.generate(
            src_tensor,
            max_len=max_len,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx
        )
    
    # Decode output
    output_tokens = tgt_vocab.decode(output_indices[0].tolist())
    
    # Remove special tokens and join
    translation = ' '.join([
        token for token in output_tokens 
        if token not in [tgt_vocab.PAD_TOKEN, tgt_vocab.BOS_TOKEN, tgt_vocab.EOS_TOKEN]
    ])
    
    return translation


def main(args):
    device = Config.device
    print(f"Using device: {device}")
    
    # Load vocabularies
    print("Loading vocabularies...")
    src_vocab = Vocabulary.load(os.path.join(Config.save_dir, 'src_vocab.pkl'))
    tgt_vocab = Vocabulary.load(os.path.join(Config.save_dir, 'tgt_vocab.pkl'))
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Build model
    print("Building model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        n_encoder_layers=Config.n_encoder_layers,
        n_decoder_layers=Config.n_decoder_layers,
        d_ff=Config.d_ff,
        max_seq_len=Config.max_seq_len,
        dropout=0.0,  # No dropout for evaluation
        pad_idx=Config.pad_idx
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['loss']:.4f}")
    
    # Interactive translation or batch
    if args.interactive:
        print("\n" + "="*60)
        print("Interactive Translation (type 'quit' to exit)")
        print("="*60)
        
        while True:
            src_text = input("\nEnglish: ").strip()
            if src_text.lower() == 'quit':
                break
            
            if not src_text:
                continue
            
            translation = translate(model, src_text, src_vocab, tgt_vocab, device, args.max_len)
            print(f"German:  {translation}")
    
    elif args.use_test_set and args.dataset == 'multi30k':
        # Load test set from Multi30K
        print("\n" + "="*60)
        print("Evaluating on Multi30K Test Set")
        print("="*60)
        
        test_en_path = os.path.join(Config.data_dir, 'test', 'test2016.en')
        test_de_path = os.path.join(Config.data_dir, 'test', 'test2016.de')
        
        with open(test_en_path, 'r', encoding='utf-8') as f:
            test_en = [line.strip() for line in f if line.strip()]
        
        with open(test_de_path, 'r', encoding='utf-8') as f:
            test_de = [line.strip() for line in f if line.strip()]
        
        # Determine number of samples to evaluate
        if args.num_samples == -1:
            num_samples = len(test_en)
        else:
            num_samples = min(args.num_samples, len(test_en))
        
        print(f"Translating {num_samples} test samples...")
        
        predictions = []
        references = []
        
        for i in range(num_samples):
            translation = translate(model, test_en[i], src_vocab, tgt_vocab, device, args.max_len)
            predictions.append(translation)
            references.append(test_de[i])
            
            if args.verbose and i < 10:  # Show first 10 examples if verbose
                print(f"\n[{i+1}/{num_samples}]")
                print(f"EN: {test_en[i]}")
                print(f"DE (ref): {test_de[i]}")
                print(f"DE (pred): {translation}")
        
        # Calculate BLEU score
        print("\n" + "="*60)
        print("BLEU Scores")
        print("="*60)
        
        bleu_scores = compute_bleu(predictions, references)
        
        if "error" in bleu_scores:
            print(f"Error: {bleu_scores['error']}")
        else:
            print(f"BLEU Score: {bleu_scores['bleu']:.2f}")
            print(f"BLEU-1: {bleu_scores['bleu_1']:.2f}")
            print(f"BLEU-2: {bleu_scores['bleu_2']:.2f}")
            print(f"BLEU-3: {bleu_scores['bleu_3']:.2f}")
            print(f"BLEU-4: {bleu_scores['bleu_4']:.2f}")
    
    else:
        # Default test examples
        test_sentences = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love learning new languages.",
            "Thank you very much for your help.",
            "What time is it?"
        ]
        
        print("\n" + "="*60)
        print("Translation Examples")
        print("="*60)
        
        for src_text in test_sentences:
            translation = translate(model, src_text, src_vocab, tgt_vocab, device, args.max_len)
            print(f"\nEN: {src_text}")
            print(f"DE: {translation}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    parser.add_argument('--dataset', type=str, default='multi30k',
                       choices=['iwslt', 'multi30k'],
                       help='Dataset to use (iwslt or multi30k)')
    parser.add_argument('--checkpoint', type=str, 
                       default='results/checkpoints/checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum generation length')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive translation mode')
    parser.add_argument('--use-test-set', action='store_true',
                       help='Evaluate on actual test set (Multi30K only)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of test samples to evaluate (-1 for all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed translation examples')
    
    args = parser.parse_args()
    main(args)


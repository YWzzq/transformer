echo "=========================================="
echo "Transformer Training and Ablation Study"
echo "=========================================="

# Set random seed for reproducibility
export PYTHONHASHSEED=42

# Step 1: Full ablation study (includes baseline + all variants)
echo ""
echo "Step 1: Ablation study on Multi30k (includes baseline)"
echo "-------------------------------------------------------"
echo "This will train 7 configurations:"
echo "  - Baseline (8 heads, 4 layers, FFN 1536, Dropout 0.25)"
echo "  - 4 Heads, 2 Heads"
echo "  - 2 Layers"
echo "  - FFN 512"
echo "  - No Dropout"
echo "  - No Positional Encoding"
echo ""
echo
    python scripts/ablation_study.py  --epochs 10 --run-all

# Step 2: Evaluation on test set
echo ""
echo "Step 2: Evaluation on Multi30k test set"
echo "----------------------------------------"
if [ -f "results/checkpoints/ablation_Baseline.pth" ]; then
    echo "Evaluating baseline model..."
    python scripts/evaluate.py \
        --checkpoint results/checkpoints/ablation_Baseline.pth \
        --dataset multi30k \
        --use-test-set \
        --verbose
else
    echo "No baseline checkpoint found. Please run ablation study first."
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in results/"
echo "=========================================="


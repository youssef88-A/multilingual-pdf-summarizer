"""
ONNX/TensorRT conversion for model optimization.
Shows deployment-oriented optimization work.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import onnx
import onnxruntime
from onnxruntime.transformers import optimizer
import time
import psutil
import os

def convert_to_onnx():
    """Convert mT5 model to ONNX format for faster inference."""
    
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # Prepare dummy input
    dummy_text = "summarize: This is a test input for ONNX conversion."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "models/mt5_multilingual.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print("ONNX export complete!")
    
    # Verify ONNX model
    onnx_model = onnx.load("models/mt5_multilingual.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified!")
    
    # Optimize ONNX model
    print("Optimizing ONNX model...")
    optimized_model = optimizer.optimize_model(
        "models/mt5_multilingual.onnx",
        model_type='bert',
        num_heads=12,
        hidden_size=768
    )
    optimized_model.save_model_to_file("models/mt5_multilingual_optimized.onnx")
    
    return "models/mt5_multilingual_optimized.onnx"

def benchmark_inference():
    """Benchmark inference speed between PyTorch and ONNX."""
    
    # Load original PyTorch model
    print("\nBenchmarking PyTorch model...")
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pt_model.eval()
    
    # Load ONNX model
    print("Loading ONNX model...")
    ort_session = onnxruntime.InferenceSession("models/mt5_multilingual_optimized.onnx")
    
    # Prepare test inputs
    test_texts = [
        "This is a short test.",
        "This is a medium length test with multiple sentences. It should be processed efficiently.",
        "This is a longer test with multiple sentences. " * 10
    ]
    
    results = {}
    
    for i, text in enumerate(test_texts):
        inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
        
        # PyTorch inference
        start_time = time.time()
        with torch.no_grad():
            pt_outputs = pt_model.generate(
                inputs['input_ids'],
                max_length=100,
                num_beams=4
            )
        pt_time = time.time() - start_time
        
        # ONNX inference
        ort_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
        start_time = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_time = time.time() - start_time
        
        # Calculate speedup
        speedup = pt_time / ort_time
        
        results[f"text_{i+1}"] = {
            "pt_time": pt_time,
            "ort_time": ort_time,
            "speedup": speedup,
            "length": len(text)
        }
    
    # Print results
    print("\n" + "="*60)
    print("Inference Benchmark Results")
    print("="*60)
    for key, value in results.items():
        print(f"\n{key}:")
        print(f"  Text length: {value['length']} chars")
        print(f"  PyTorch time: {value['pt_time']:.4f}s")
        print(f"  ONNX time: {value['ort_time']:.4f}s")
        print(f"  Speedup: {value['speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    # Convert model
    onnx_path = convert_to_onnx()
    print(f"\nConverted model saved to: {onnx_path}")
    
    # Run benchmarks
    benchmark_results = benchmark_inference()
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024
    print(f"\nMemory usage: {memory_usage:.2f} MB")
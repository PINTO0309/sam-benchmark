import torch
import numpy as np
import time
import onnx
import onnxruntime as ort
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import argparse
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt

def download_model_weights():
    """Download model weights if not present"""
    models = {
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    os.makedirs("weights", exist_ok=True)
    
    for model_name, url in models.items():
        weight_path = f"weights/{model_name}.pth"
        if not os.path.exists(weight_path):
            print(f"Downloading {model_name} weights...")
            os.system(f"wget -O {weight_path} {url}")
    
    return {
        "sam_vit_b": "weights/sam_vit_b.pth",
        "sam_vit_l": "weights/sam_vit_l.pth", 
        "sam_vit_h": "weights/sam_vit_h.pth"
    }

def export_sam_encoder_to_onnx(sam_model, model_type: str, output_path: str, image_size: int = 1024):
    """Export only SAM image encoder to ONNX format"""
    sam_model.eval()
    
    # Get image encoder
    image_encoder = sam_model.image_encoder
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export
    torch.onnx.export(
        image_encoder,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print(f"Exported {model_type} encoder to {output_path}")

def benchmark_pytorch_sam(model_type: str, model_path: str, image_path: str, device: str = "cuda") -> Dict:
    """Benchmark PyTorch SAM model"""
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    warmup_runs = 3
    benchmark_runs = 10
    
    # Warmup
    for _ in range(warmup_runs):
        predictor.set_image(image)
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
    
    # Benchmark image encoding only
    encoding_times = []
    for _ in range(benchmark_runs):
        predictor.reset_image()
        start = time.time()
        predictor.set_image(image)
        end = time.time()
        encoding_times.append(end - start)
    
    # Benchmark full prediction
    predictor.set_image(image)
    prediction_times = []
    for _ in range(benchmark_runs):
        start = time.time()
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        end = time.time()
        prediction_times.append(end - start)
    
    return {
        "model": f"SAM {model_type.upper()}",
        "backend": "PyTorch",
        "device": device,
        "encoding_mean_time": np.mean(encoding_times),
        "encoding_std_time": np.std(encoding_times),
        "prediction_mean_time": np.mean(prediction_times),
        "prediction_std_time": np.std(prediction_times),
        "total_mean_time": np.mean(encoding_times) + np.mean(prediction_times),
        "total_std_time": np.sqrt(np.var(encoding_times) + np.var(prediction_times))
    }

def benchmark_onnx_encoder(onnx_path: str, image_path: str, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) -> Dict:
    """Benchmark ONNX encoder model"""
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))
    
    # Normalize image
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = (image - pixel_mean.astype(np.float32)) / pixel_std.astype(np.float32)
    
    warmup_runs = 3
    benchmark_runs = 10
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, {"image": image})
    
    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start = time.time()
        _ = session.run(None, {"image": image})
        end = time.time()
        times.append(end - start)
    
    return {
        "model": os.path.basename(onnx_path).replace('.onnx', '').upper(),
        "backend": "ONNX Runtime",
        "device": providers[0].replace("ExecutionProvider", ""),
        "encoding_mean_time": np.mean(times),
        "encoding_std_time": np.std(times),
        "prediction_mean_time": 0,  # ONNX export only includes encoder
        "prediction_std_time": 0,
        "total_mean_time": np.mean(times),
        "total_std_time": np.std(times)
    }

def create_sample_image(path: str = "sample_image.jpg", size: int = 1024):
    """Create a sample image for testing"""
    # Create a more realistic test image with some structure
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add background
    img[:] = [50, 100, 150]
    
    # Add some rectangles
    cv2.rectangle(img, (100, 100), (400, 400), (255, 100, 100), -1)
    cv2.rectangle(img, (600, 200), (900, 500), (100, 255, 100), -1)
    cv2.rectangle(img, (200, 600), (500, 900), (100, 100, 255), -1)
    
    # Add some circles
    cv2.circle(img, (700, 700), 150, (255, 255, 100), -1)
    cv2.circle(img, (300, 300), 80, (255, 100, 255), -1)
    
    cv2.imwrite(path, img)
    return path

def visualize_results(results: List[Dict], output_path: str = "benchmark_results.png"):
    """Visualize benchmark results"""
    models = [r["model"] for r in results]
    backends = [r["backend"] for r in results]
    encoding_times = [r["encoding_mean_time"] for r in results]
    encoding_stds = [r["encoding_std_time"] for r in results]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars = ax.bar(x, encoding_times, width, yerr=encoding_stds, capsize=10)
    
    # Color bars by backend
    colors = ['blue' if 'PyTorch' in b else 'green' for b in backends]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Encoding Time (seconds)')
    ax.set_title('SAM Model Encoding Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({b})" for m, b in zip(models, backends)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark SAM models (lightweight to heavy)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--models", type=str, nargs='+', default=["vit_b"], choices=["vit_b", "vit_l", "vit_h"], help="Models to benchmark")
    args = parser.parse_args()
    
    # Create sample image if not provided
    if args.image is None:
        args.image = create_sample_image()
        print(f"Created sample image: {args.image}")
    
    # Download model weights
    model_paths = download_model_weights()
    
    results = []
    
    # Benchmark PyTorch models
    for model_type in args.models:
        model_path = model_paths[f"sam_{model_type}"]
        if not os.path.exists(model_path):
            print(f"Skipping {model_type} - weights not found")
            continue
            
        print(f"\nBenchmarking PyTorch SAM {model_type.upper()}...")
        try:
            result = benchmark_pytorch_sam(model_type, model_path, args.image, args.device)
            results.append(result)
            print(f"SAM {model_type.upper()} PyTorch:")
            print(f"  Encoding: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            print(f"  Prediction: {result['prediction_mean_time']:.4f}s ± {result['prediction_std_time']:.4f}s")
            print(f"  Total: {result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s")
        except Exception as e:
            print(f"Error benchmarking PyTorch SAM {model_type}: {e}")
    
    # Export to ONNX and benchmark
    print("\nExporting models to ONNX...")
    os.makedirs("onnx_models", exist_ok=True)
    
    for model_type in args.models:
        model_path = model_paths[f"sam_{model_type}"]
        if not os.path.exists(model_path):
            continue
            
        try:
            sam = sam_model_registry[model_type](checkpoint=model_path)
            onnx_path = f"onnx_models/sam_{model_type}_encoder.onnx"
            export_sam_encoder_to_onnx(sam, model_type, onnx_path)
            
            print(f"\nBenchmarking ONNX SAM {model_type.upper()} encoder...")
            if args.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            result = benchmark_onnx_encoder(onnx_path, args.image, providers)
            results.append(result)
            print(f"SAM {model_type.upper()} ONNX Encoder: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
        except Exception as e:
            print(f"Error with ONNX SAM {model_type}: {e}")
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    if len(results) > 0:
        visualize_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Backend':<15} {'Encoding Time':<20} {'Total Time':<20}")
    print("-"*60)
    for result in results:
        model = result['model']
        backend = result['backend']
        enc_time = f"{result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s"
        total_time = f"{result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s"
        print(f"{model:<20} {backend:<15} {enc_time:<20} {total_time:<20}")

if __name__ == "__main__":
    main()
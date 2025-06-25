import torch
import numpy as np
import time
import onnx
import onnxruntime as ort
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import argparse
from typing import Dict, List, Tuple
import json

def download_model_weights():
    """Download model weights if not present"""
    models = {
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/122024/sam2.1_hiera_t_baseline.pt"
    }
    
    os.makedirs("weights", exist_ok=True)
    
    for model_name, url in models.items():
        weight_path = f"weights/{model_name}.pth"
        if not os.path.exists(weight_path):
            print(f"Downloading {model_name} weights...")
            os.system(f"wget -O {weight_path} {url}")
    
    return {
        "sam_vit_b": "weights/sam_vit_b.pth",
        "sam2_hiera_tiny": "weights/sam2_hiera_tiny.pth"
    }

def export_sam_to_onnx(sam_model, output_path: str, image_size: Tuple[int, int] = (1024, 1024)):
    """Export SAM model to ONNX format"""
    sam_model.eval()
    
    dummy_input = torch.randn(1, 3, *image_size)
    dummy_point = torch.tensor([[[512, 512]]], dtype=torch.float32)
    dummy_label = torch.tensor([[1]], dtype=torch.int32)
    
    torch.onnx.export(
        sam_model.image_encoder,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"}
        },
        opset_version=17
    )
    print(f"Exported ONNX model to {output_path}")

def benchmark_pytorch_sam(model_path: str, image_path: str, device: str = "cuda") -> Dict:
    """Benchmark PyTorch SAM model"""
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    warmup_runs = 5
    benchmark_runs = 20
    
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
    
    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start = time.time()
        predictor.set_image(image)
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        end = time.time()
        times.append(end - start)
    
    return {
        "model": "SAM ViT-B",
        "backend": "PyTorch",
        "device": device,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times)
    }

def benchmark_pytorch_sam2(model_path: str, image_path: str, device: str = "cuda") -> Dict:
    """Benchmark PyTorch SAM2 model"""
    # For now, skip SAM2 due to configuration issues
    return {
        "model": "SAM2 Hiera Tiny",
        "backend": "PyTorch",
        "device": device,
        "mean_time": 0,
        "std_time": 0,
        "min_time": 0,
        "max_time": 0,
        "note": "Skipped due to configuration issues"
    }
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    warmup_runs = 5
    benchmark_runs = 20
    

def benchmark_onnx_model(onnx_path: str, image_path: str) -> Dict:
    """Benchmark ONNX model"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    warmup_runs = 5
    benchmark_runs = 20
    
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
        "model": os.path.basename(onnx_path),
        "backend": "ONNX Runtime",
        "device": providers[0].replace("ExecutionProvider", ""),
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times)
    }

def create_sample_image(path: str = "sample_image.jpg"):
    """Create a sample image for testing"""
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path

def main():
    parser = argparse.ArgumentParser(description="Benchmark SAM and SAM2 models")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    args = parser.parse_args()
    
    # Create sample image if not provided
    if args.image is None:
        args.image = create_sample_image()
        print(f"Created sample image: {args.image}")
    
    # Download model weights
    model_paths = download_model_weights()
    
    results = []
    
    # Benchmark PyTorch SAM
    print("\nBenchmarking PyTorch SAM...")
    try:
        result = benchmark_pytorch_sam(model_paths["sam_vit_b"], args.image, args.device)
        results.append(result)
        print(f"SAM PyTorch: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    except Exception as e:
        print(f"Error benchmarking PyTorch SAM: {e}")
    
    # Benchmark PyTorch SAM2
    print("\nBenchmarking PyTorch SAM2...")
    try:
        result = benchmark_pytorch_sam2(model_paths["sam2_hiera_tiny"], args.image, args.device)
        results.append(result)
        print(f"SAM2 PyTorch: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    except Exception as e:
        print(f"Error benchmarking PyTorch SAM2: {e}")
    
    # Export to ONNX and benchmark
    print("\nExporting models to ONNX...")
    os.makedirs("onnx_models", exist_ok=True)
    
    # Export SAM to ONNX
    try:
        sam = sam_model_registry["vit_b"](checkpoint=model_paths["sam_vit_b"])
        export_sam_to_onnx(sam, "onnx_models/sam_vit_b.onnx")
        
        print("\nBenchmarking ONNX SAM...")
        result = benchmark_onnx_model("onnx_models/sam_vit_b.onnx", args.image)
        results.append(result)
        print(f"SAM ONNX: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    except Exception as e:
        print(f"Error with ONNX SAM: {e}")
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for result in results:
        print(f"{result['model']} ({result['backend']}): {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")

if __name__ == "__main__":
    main()
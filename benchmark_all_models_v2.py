import torch
import numpy as np
import time
import onnx
import onnxruntime as ort
from PIL import Image
import cv2
import os
import argparse
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import gdown

# Original SAM
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Lightning SAM support
try:
    from lightning_sam.model import Model as LightningSAMModel
    from lightning_sam.config import cfg as lightning_cfg
    LIGHTNING_SAM_AVAILABLE = True
except ImportError:
    LIGHTNING_SAM_AVAILABLE = False
    print("Warning: Lightning SAM not available. Install from https://github.com/luca-medeiros/lightning-sam")

# MobileSAM support
try:
    from mobile_sam import sam_model_registry as mobile_sam_registry
    from mobile_sam import SamPredictor as MobileSamPredictor
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False
    print("Warning: MobileSAM not available. Install from https://github.com/ChaoningZhang/MobileSAM")

# TinySAM support
try:
    sys.path.append('tinysam')
    from tinysam import sam_model_registry as tiny_sam_registry
    from tinysam import SamPredictor as TinySamPredictor
    TINYSAM_AVAILABLE = True
except ImportError:
    TINYSAM_AVAILABLE = False
    print("Warning: TinySAM not available. Clone from https://github.com/xinghaochen/TinySAM")

# SlimSAM support
SLIMSAM_AVAILABLE = True  # Uses standard segment_anything

# Expedit-SAM support
try:
    from segment_anything import build_sam
    EXPEDIT_SAM_AVAILABLE = True
except ImportError:
    EXPEDIT_SAM_AVAILABLE = False
    print("Warning: Expedit-SAM requires modified segment_anything")

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark"""
    model_type: str
    implementation: str  # 'pytorch', 'onnx', 'lightning', 'mobile', 'tiny', 'slim', 'expedit'
    device: str
    image_size: int = 1024
    warmup_runs: int = 3
    benchmark_runs: int = 10

def download_model_weights():
    """Download model weights if not present"""
    models = {
        # Original SAM
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        # MobileSAM
        "mobile_sam": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        # TinySAM
        "tinysam": "https://github.com/xinghaochen/TinySAM/releases/download/3.0/tinysam_42.3.pth",
        "tinysam_q": "https://github.com/xinghaochen/TinySAM/releases/download/2.0/tinysam_w8a8.pth",
    }
    
    # SlimSAM weights (Google Drive - manual download required)
    slim_models = {
        "slimsam_50": "1bTjBZs2oWHeo6OPxumD_Gces4VCcU0JI",  # SlimSAM-50
        "slimsam_77": "14BhU66umvY0E1FWoGsCMpLqXMNw9c3Nx",  # SlimSAM-77
        "slimsam_50_uniform": "1Ld7Q2LY8H2nu4zB6VxwwA5npS5A9OHFq",  # SlimSAM-50-uniform
        "slimsam_77_uniform": "1OeWpfk5WhdlMz5VvYmb9gaE6suzHB0sp",  # SlimSAM-77-uniform
    }
    
    os.makedirs("weights", exist_ok=True)
    
    # Download direct links
    for model_name, url in models.items():
        if "mobile_sam" in model_name:
            weight_path = f"weights/{model_name}.pt"
        else:
            weight_path = f"weights/{model_name}.pth"
            
        if not os.path.exists(weight_path):
            print(f"Downloading {model_name} weights...")
            os.system(f"wget -O {weight_path} {url}")
    
    # Handle Google Drive downloads for SlimSAM
    for model_name, file_id in slim_models.items():
        weight_path = f"weights/{model_name}.pth"
        if not os.path.exists(weight_path):
            print(f"Downloading {model_name} from Google Drive...")
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, weight_path, quiet=False)
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                print(f"Please download manually from: https://drive.google.com/file/d/{file_id}/view")
    
    result = {
        "sam_vit_b": "weights/sam_vit_b.pth",
        "sam_vit_l": "weights/sam_vit_l.pth", 
        "sam_vit_h": "weights/sam_vit_h.pth",
        "mobile_sam": "weights/mobile_sam.pt",
        "tinysam": "weights/tinysam.pth",
        "tinysam_q": "weights/tinysam_q.pth",
        "slimsam_50": "weights/slimsam_50.pth",
        "slimsam_77": "weights/slimsam_77.pth",
        "slimsam_50_uniform": "weights/slimsam_50_uniform.pth",
        "slimsam_77_uniform": "weights/slimsam_77_uniform.pth",
    }
    
    return result

def benchmark_model(predictor, image_path: str, config: BenchmarkConfig, use_multimask=True) -> Dict:
    """Generic benchmark function for SAM-like models"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Warmup
    for _ in range(config.warmup_runs):
        predictor.set_image(image)
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        if use_multimask:
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        else:
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
            )
    
    # Benchmark image encoding
    encoding_times = []
    for _ in range(config.benchmark_runs):
        predictor.reset_image()
        start = time.time()
        predictor.set_image(image)
        end = time.time()
        encoding_times.append(end - start)
    
    # Benchmark full prediction
    predictor.set_image(image)
    prediction_times = []
    for _ in range(config.benchmark_runs):
        start = time.time()
        input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
        input_label = np.array([1])
        if use_multimask:
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        else:
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
            )
        end = time.time()
        prediction_times.append(end - start)
    
    return {
        "encoding_mean_time": np.mean(encoding_times),
        "encoding_std_time": np.std(encoding_times),
        "prediction_mean_time": np.mean(prediction_times),
        "prediction_std_time": np.std(prediction_times),
        "total_mean_time": np.mean(encoding_times) + np.mean(prediction_times),
        "total_std_time": np.sqrt(np.var(encoding_times) + np.var(prediction_times))
    }

def benchmark_pytorch_sam(config: BenchmarkConfig, model_path: str, image_path: str) -> Dict:
    """Benchmark PyTorch SAM model"""
    sam = sam_model_registry[config.model_type](checkpoint=model_path)
    sam.to(device=config.device)
    predictor = SamPredictor(sam)
    
    result = benchmark_model(predictor, image_path, config)
    result.update({
        "model": f"SAM {config.model_type.upper()}",
        "implementation": "PyTorch",
        "device": config.device,
    })
    return result

def benchmark_mobile_sam(config: BenchmarkConfig, model_path: str, image_path: str) -> Dict:
    """Benchmark MobileSAM model"""
    if not MOBILE_SAM_AVAILABLE:
        return {
            "model": "MobileSAM",
            "implementation": "MobileSAM",
            "device": config.device,
            "error": "MobileSAM not available"
        }
    
    try:
        mobile_sam = mobile_sam_registry["vit_t"](checkpoint=model_path)
        mobile_sam.to(device=config.device)
        mobile_sam.eval()
        predictor = MobileSamPredictor(mobile_sam)
        
        result = benchmark_model(predictor, image_path, config)
        result.update({
            "model": "MobileSAM",
            "implementation": "MobileSAM",
            "device": config.device,
        })
        return result
    except Exception as e:
        return {
            "model": "MobileSAM",
            "implementation": "MobileSAM",
            "device": config.device,
            "error": str(e)
        }

def benchmark_tiny_sam(config: BenchmarkConfig, model_path: str, image_path: str) -> Dict:
    """Benchmark TinySAM model"""
    if not TINYSAM_AVAILABLE:
        return {
            "model": "TinySAM",
            "implementation": "TinySAM",
            "device": config.device,
            "error": "TinySAM not available"
        }
    
    try:
        tiny_sam = tiny_sam_registry["vit_t"](checkpoint=model_path)
        tiny_sam.to(device=config.device)
        tiny_sam.eval()
        predictor = TinySamPredictor(tiny_sam)
        
        result = benchmark_model(predictor, image_path, config, use_multimask=False)
        result.update({
            "model": "TinySAM",
            "implementation": "TinySAM",
            "device": config.device,
        })
        return result
    except Exception as e:
        return {
            "model": "TinySAM",
            "implementation": "TinySAM",
            "device": config.device,
            "error": str(e)
        }

def benchmark_slim_sam(config: BenchmarkConfig, model_path: str, image_path: str, model_variant: str) -> Dict:
    """Benchmark SlimSAM model"""
    try:
        if "uniform" in model_variant:
            # Local pruning models (recommended)
            if "50" in model_variant:
                model_type = 'vit_p50'
            else:
                model_type = 'vit_p77'
            slim_sam = sam_model_registry[model_type](checkpoint=model_path)
        else:
            # Global pruning models
            slim_sam = torch.load(model_path)
            if hasattr(slim_sam.image_encoder, 'module'):
                slim_sam.image_encoder = slim_sam.image_encoder.module
        
        slim_sam.to(device=config.device)
        slim_sam.eval()
        predictor = SamPredictor(slim_sam)
        
        result = benchmark_model(predictor, image_path, config)
        result.update({
            "model": f"SlimSAM-{model_variant}",
            "implementation": "SlimSAM",
            "device": config.device,
        })
        return result
    except Exception as e:
        return {
            "model": f"SlimSAM-{model_variant}",
            "implementation": "SlimSAM",
            "device": config.device,
            "error": str(e)
        }

def benchmark_expedit_sam(config: BenchmarkConfig, model_path: str, image_path: str, 
                         clustering_location: int = 6, num_cluster: int = 81) -> Dict:
    """Benchmark Expedit-SAM model"""
    if not EXPEDIT_SAM_AVAILABLE:
        return {
            "model": f"Expedit-SAM {config.model_type.upper()}",
            "implementation": "Expedit-SAM",
            "device": config.device,
            "error": "Expedit-SAM not available"
        }
    
    try:
        # Build SAM with hourglass acceleration
        sam = build_sam(
            checkpoint=model_path,
            use_hourglass=True,
            hourglass_clustering_location=clustering_location,
            hourglass_num_cluster=num_cluster
        )
        sam.to(device=config.device)
        predictor = SamPredictor(sam)
        
        result = benchmark_model(predictor, image_path, config)
        result.update({
            "model": f"Expedit-SAM {config.model_type.upper()} (loc={clustering_location}, n={num_cluster})",
            "implementation": "Expedit-SAM",
            "device": config.device,
        })
        return result
    except Exception as e:
        return {
            "model": f"Expedit-SAM {config.model_type.upper()}",
            "implementation": "Expedit-SAM",
            "device": config.device,
            "error": str(e)
        }

def create_sample_image(path: str = "sample_image.jpg", size: int = 1024):
    """Create a sample image for testing"""
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
    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    models = [f"{r['model']}\n{r['implementation']}" for r in valid_results]
    encoding_times = [r["encoding_mean_time"] for r in valid_results]
    encoding_stds = [r["encoding_std_time"] for r in valid_results]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.7
    
    bars = ax.bar(x, encoding_times, width, yerr=encoding_stds, capsize=10)
    
    # Color bars by implementation
    colors = {
        'PyTorch': 'blue', 
        'ONNX Runtime': 'green', 
        'Lightning': 'orange', 
        'MobileSAM': 'red',
        'TinySAM': 'purple',
        'SlimSAM': 'brown',
        'Expedit-SAM': 'pink'
    }
    for bar, result in zip(bars, valid_results):
        bar.set_color(colors.get(result['implementation'], 'gray'))
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Encoding Time (seconds)', fontsize=12)
    ax.set_title('SAM Model Variants Encoding Benchmark Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=impl) 
                      for impl, color in colors.items() if any(r['implementation'] == impl for r in valid_results)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add value labels on bars
    for bar, result in zip(bars, valid_results):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive SAM variants benchmark")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=["vit_b", "mobile", "tiny", "slim50", "expedit"],
                       help="Models to benchmark: vit_b, vit_l, vit_h, mobile, tiny, slim50, slim77, expedit")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer runs")
    args = parser.parse_args()
    
    # Create sample image if not provided
    if args.image is None:
        args.image = create_sample_image()
        print(f"Created sample image: {args.image}")
    
    # Download model weights
    model_paths = download_model_weights()
    
    results = []
    
    # Adjust runs for quick mode
    if args.quick:
        warmup_runs = 1
        benchmark_runs = 3
    else:
        warmup_runs = 3
        benchmark_runs = 10
    
    for model in args.models:
        # Original SAM models
        if model in ["vit_b", "vit_l", "vit_h"]:
            model_path = model_paths[f"sam_{model}"]
            if not os.path.exists(model_path):
                print(f"Skipping SAM {model.upper()} - weights not found")
                continue
                
            print(f"\nBenchmarking PyTorch SAM {model.upper()}...")
            try:
                config = BenchmarkConfig(
                    model_type=model,
                    implementation="pytorch",
                    device=args.device,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs
                )
                result = benchmark_pytorch_sam(config, model_path, args.image)
                results.append(result)
                print(f"SAM {model.upper()} PyTorch: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            except Exception as e:
                print(f"Error benchmarking PyTorch SAM {model}: {e}")
        
        # MobileSAM
        elif model == "mobile":
            model_path = model_paths["mobile_sam"]
            if not os.path.exists(model_path):
                print(f"Skipping MobileSAM - weights not found")
                continue
                
            print(f"\nBenchmarking MobileSAM...")
            config = BenchmarkConfig(
                model_type="mobile",
                implementation="mobile",
                device=args.device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs
            )
            result = benchmark_mobile_sam(config, model_path, args.image)
            results.append(result)
            if "error" not in result:
                print(f"MobileSAM: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            else:
                print(f"MobileSAM Error: {result['error']}")
        
        # TinySAM
        elif model == "tiny":
            model_path = model_paths["tinysam"]
            if not os.path.exists(model_path):
                print(f"Skipping TinySAM - weights not found")
                continue
                
            print(f"\nBenchmarking TinySAM...")
            config = BenchmarkConfig(
                model_type="tiny",
                implementation="tiny",
                device=args.device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs
            )
            result = benchmark_tiny_sam(config, model_path, args.image)
            results.append(result)
            if "error" not in result:
                print(f"TinySAM: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            else:
                print(f"TinySAM Error: {result['error']}")
        
        # SlimSAM variants
        elif model.startswith("slim"):
            variant = model[4:]  # Remove "slim" prefix
            if variant == "50":
                model_path = model_paths["slimsam_50_uniform"]
                variant_name = "50-uniform"
            elif variant == "77":
                model_path = model_paths["slimsam_77_uniform"]
                variant_name = "77-uniform"
            else:
                continue
                
            if not os.path.exists(model_path):
                print(f"Skipping SlimSAM-{variant} - weights not found")
                continue
                
            print(f"\nBenchmarking SlimSAM-{variant}...")
            config = BenchmarkConfig(
                model_type=f"slim{variant}",
                implementation="slim",
                device=args.device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs
            )
            result = benchmark_slim_sam(config, model_path, args.image, variant_name)
            results.append(result)
            if "error" not in result:
                print(f"SlimSAM-{variant}: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            else:
                print(f"SlimSAM-{variant} Error: {result['error']}")
        
        # Expedit-SAM
        elif model == "expedit":
            # Test with SAM ViT-B as base
            model_path = model_paths["sam_vit_b"]
            if not os.path.exists(model_path):
                print(f"Skipping Expedit-SAM - base weights not found")
                continue
                
            print(f"\nBenchmarking Expedit-SAM (1.5x speedup config)...")
            config = BenchmarkConfig(
                model_type="vit_b",
                implementation="expedit",
                device=args.device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs
            )
            result = benchmark_expedit_sam(config, model_path, args.image, 
                                         clustering_location=6, num_cluster=81)
            results.append(result)
            if "error" not in result:
                print(f"Expedit-SAM: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
            else:
                print(f"Expedit-SAM Error: {result['error']}")
    
    # Save results
    with open("benchmark_results_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    if len(results) > 0:
        visualize_results(results, "benchmark_results_comprehensive.png")
    
    # Print summary
    print("\n" + "="*100)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Model':<35} {'Implementation':<15} {'Encoding Time':<20} {'Total Time':<20}")
    print("-"*100)
    for result in results:
        if "error" in result:
            print(f"{result['model']:<35} {result['implementation']:<15} ERROR: {result['error']}")
        else:
            model = result['model']
            impl = result['implementation']
            enc_time = f"{result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s"
            total_time = f"{result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s"
            print(f"{model:<35} {impl:<15} {enc_time:<20} {total_time:<20}")

if __name__ == "__main__":
    main()
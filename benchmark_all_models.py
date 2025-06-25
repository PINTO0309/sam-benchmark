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
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark"""
    model_type: str
    implementation: str  # 'pytorch', 'onnx', 'lightning', 'mobile'
    device: str
    image_size: int = 1024
    warmup_runs: int = 3
    benchmark_runs: int = 10

def download_model_weights():
    """Download model weights if not present"""
    models = {
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "mobile_sam": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    }
    
    os.makedirs("weights", exist_ok=True)
    
    for model_name, url in models.items():
        weight_path = f"weights/{model_name}.pth" if model_name != "mobile_sam" else "weights/mobile_sam.pt"
        if not os.path.exists(weight_path):
            print(f"Downloading {model_name} weights...")
            os.system(f"wget -O {weight_path} {url}")
    
    return {
        "sam_vit_b": "weights/sam_vit_b.pth",
        "sam_vit_l": "weights/sam_vit_l.pth", 
        "sam_vit_h": "weights/sam_vit_h.pth",
        "mobile_sam": "weights/mobile_sam.pt"
    }

def create_lightning_sam_model(model_type: str, checkpoint_path: str, device: str = "cuda"):
    """Create Lightning SAM model"""
    if not LIGHTNING_SAM_AVAILABLE:
        raise ImportError("Lightning SAM not available")
    
    # Simple config setup
    class SimpleConfig:
        def __init__(self):
            self.model = type('obj', (object,), {
                'type': model_type,
                'checkpoint': checkpoint_path,
                'freeze': type('obj', (object,), {
                    'image_encoder': False,
                    'prompt_encoder': False,
                    'mask_decoder': False
                })
            })
    
    cfg = SimpleConfig()
    model = LightningSAMModel(cfg)
    model.setup()
    model.to(device)
    model.eval()
    return model

def benchmark_pytorch_sam(config: BenchmarkConfig, model_path: str, image_path: str) -> Dict:
    """Benchmark PyTorch SAM model"""
    sam = sam_model_registry[config.model_type](checkpoint=model_path)
    sam.to(device=config.device)
    predictor = SamPredictor(sam)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Warmup
    for _ in range(config.warmup_runs):
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
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        end = time.time()
        prediction_times.append(end - start)
    
    return {
        "model": f"SAM {config.model_type.upper()}",
        "implementation": "PyTorch",
        "device": config.device,
        "encoding_mean_time": np.mean(encoding_times),
        "encoding_std_time": np.std(encoding_times),
        "prediction_mean_time": np.mean(prediction_times),
        "prediction_std_time": np.std(prediction_times),
        "total_mean_time": np.mean(encoding_times) + np.mean(prediction_times),
        "total_std_time": np.sqrt(np.var(encoding_times) + np.var(prediction_times))
    }

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
        # MobileSAM uses vit_t
        mobile_sam = mobile_sam_registry["vit_t"](checkpoint=model_path)
        mobile_sam.to(device=config.device)
        mobile_sam.eval()
        predictor = MobileSamPredictor(mobile_sam)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Warmup
        for _ in range(config.warmup_runs):
            predictor.set_image(image)
            input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
            input_label = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
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
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            end = time.time()
            prediction_times.append(end - start)
        
        return {
            "model": "MobileSAM",
            "implementation": "MobileSAM",
            "device": config.device,
            "encoding_mean_time": np.mean(encoding_times),
            "encoding_std_time": np.std(encoding_times),
            "prediction_mean_time": np.mean(prediction_times),
            "prediction_std_time": np.std(prediction_times),
            "total_mean_time": np.mean(encoding_times) + np.mean(prediction_times),
            "total_std_time": np.sqrt(np.var(encoding_times) + np.var(prediction_times))
        }
    except Exception as e:
        return {
            "model": "MobileSAM",
            "implementation": "MobileSAM",
            "device": config.device,
            "error": str(e)
        }

def benchmark_lightning_sam(config: BenchmarkConfig, model_path: str, image_path: str) -> Dict:
    """Benchmark Lightning SAM model"""
    if not LIGHTNING_SAM_AVAILABLE:
        return {
            "model": f"Lightning SAM {config.model_type.upper()}",
            "implementation": "Lightning",
            "device": config.device,
            "error": "Lightning SAM not available"
        }
    
    try:
        model = create_lightning_sam_model(config.model_type, model_path, config.device)
        predictor = model.get_predictor()
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Warmup
        for _ in range(config.warmup_runs):
            predictor.set_image(image)
            input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
            input_label = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
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
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            end = time.time()
            prediction_times.append(end - start)
        
        return {
            "model": f"Lightning SAM {config.model_type.upper()}",
            "implementation": "Lightning",
            "device": config.device,
            "encoding_mean_time": np.mean(encoding_times),
            "encoding_std_time": np.std(encoding_times),
            "prediction_mean_time": np.mean(prediction_times),
            "prediction_std_time": np.std(prediction_times),
            "total_mean_time": np.mean(encoding_times) + np.mean(prediction_times),
            "total_std_time": np.sqrt(np.var(encoding_times) + np.var(prediction_times))
        }
    except Exception as e:
        return {
            "model": f"Lightning SAM {config.model_type.upper()}",
            "implementation": "Lightning",
            "device": config.device,
            "error": str(e)
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
        "implementation": "ONNX Runtime",
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
    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    models = [f"{r['model']}\n{r['implementation']}" for r in valid_results]
    encoding_times = [r["encoding_mean_time"] for r in valid_results]
    encoding_stds = [r["encoding_std_time"] for r in valid_results]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    bars = ax.bar(x, encoding_times, width, yerr=encoding_stds, capsize=10)
    
    # Color bars by implementation
    colors = {'PyTorch': 'blue', 'ONNX Runtime': 'green', 'Lightning': 'orange', 'MobileSAM': 'red'}
    for bar, result in zip(bars, valid_results):
        bar.set_color(colors.get(result['implementation'], 'gray'))
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Encoding Time (seconds)')
    ax.set_title('SAM Model Encoding Benchmark Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=impl) 
                      for impl, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark SAM models with multiple implementations")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--models", type=str, nargs='+', default=["vit_b"], 
                       choices=["vit_b", "vit_l", "vit_h", "mobile"], help="Models to benchmark")
    parser.add_argument("--implementations", type=str, nargs='+', 
                       default=["pytorch", "onnx", "lightning", "mobile"],
                       choices=["pytorch", "onnx", "lightning", "mobile"], 
                       help="Implementations to benchmark")
    args = parser.parse_args()
    
    # Create sample image if not provided
    if args.image is None:
        args.image = create_sample_image()
        print(f"Created sample image: {args.image}")
    
    # Download model weights
    model_paths = download_model_weights()
    
    results = []
    
    for model_type in args.models:
        # Handle MobileSAM separately
        if model_type == "mobile":
            if "mobile" in args.implementations:
                model_path = model_paths["mobile_sam"]
                if not os.path.exists(model_path):
                    print(f"Skipping MobileSAM - weights not found")
                    continue
                
                print(f"\nBenchmarking MobileSAM...")
                config = BenchmarkConfig(
                    model_type="mobile",
                    implementation="mobile",
                    device=args.device
                )
                result = benchmark_mobile_sam(config, model_path, args.image)
                results.append(result)
                if "error" in result:
                    print(f"MobileSAM Error: {result['error']}")
                else:
                    print(f"MobileSAM:")
                    print(f"  Encoding: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
                    print(f"  Prediction: {result['prediction_mean_time']:.4f}s ± {result['prediction_std_time']:.4f}s")
                    print(f"  Total: {result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s")
            continue
        
        model_path = model_paths[f"sam_{model_type}"]
        if not os.path.exists(model_path):
            print(f"Skipping {model_type} - weights not found")
            continue
        
        # PyTorch benchmark
        if "pytorch" in args.implementations:
            print(f"\nBenchmarking PyTorch SAM {model_type.upper()}...")
            try:
                config = BenchmarkConfig(
                    model_type=model_type,
                    implementation="pytorch",
                    device=args.device
                )
                result = benchmark_pytorch_sam(config, model_path, args.image)
                results.append(result)
                print(f"SAM {model_type.upper()} PyTorch:")
                print(f"  Encoding: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
                print(f"  Prediction: {result['prediction_mean_time']:.4f}s ± {result['prediction_std_time']:.4f}s")
                print(f"  Total: {result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s")
            except Exception as e:
                print(f"Error benchmarking PyTorch SAM {model_type}: {e}")
        
        # Lightning SAM benchmark
        if "lightning" in args.implementations and LIGHTNING_SAM_AVAILABLE:
            print(f"\nBenchmarking Lightning SAM {model_type.upper()}...")
            config = BenchmarkConfig(
                model_type=model_type,
                implementation="lightning",
                device=args.device
            )
            result = benchmark_lightning_sam(config, model_path, args.image)
            results.append(result)
            if "error" in result:
                print(f"Lightning SAM {model_type.upper()} Error: {result['error']}")
            else:
                print(f"Lightning SAM {model_type.upper()}:")
                print(f"  Encoding: {result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s")
                print(f"  Prediction: {result['prediction_mean_time']:.4f}s ± {result['prediction_std_time']:.4f}s")
                print(f"  Total: {result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s")
        
        # ONNX benchmark
        if "onnx" in args.implementations:
            print(f"\nExporting {model_type} to ONNX...")
            os.makedirs("onnx_models", exist_ok=True)
            try:
                sam = sam_model_registry[model_type](checkpoint=model_path)
                onnx_path = f"onnx_models/sam_{model_type}_encoder.onnx"
                export_sam_encoder_to_onnx(sam, model_type, onnx_path)
                
                print(f"Benchmarking ONNX SAM {model_type.upper()} encoder...")
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
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Implementation':<15} {'Encoding Time':<20} {'Total Time':<20}")
    print("-"*80)
    for result in results:
        if "error" in result:
            print(f"{result['model']:<25} {result['implementation']:<15} ERROR: {result['error']}")
        else:
            model = result['model']
            impl = result['implementation']
            enc_time = f"{result['encoding_mean_time']:.4f}s ± {result['encoding_std_time']:.4f}s"
            total_time = f"{result['total_mean_time']:.4f}s ± {result['total_std_time']:.4f}s"
            print(f"{model:<25} {impl:<15} {enc_time:<20} {total_time:<20}")

if __name__ == "__main__":
    main()
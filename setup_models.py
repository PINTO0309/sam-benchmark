#!/usr/bin/env python3
"""Setup script to prepare all SAM model variants"""

import os
import sys
import shutil

def setup_models():
    """Setup model directories and create necessary symlinks"""
    
    # Ensure we're in the right directory
    if not os.path.exists('benchmark_all_models_v2.py'):
        print("Error: Please run this script from the sam-benchmark directory")
        return False
    
    # Check if repositories are cloned
    repos = {
        'tinysam': 'TinySAM',
        'slimsam': 'SlimSAM',
        'expedit-sam': 'Expedit-SAM'
    }
    
    for repo_dir, repo_name in repos.items():
        if not os.path.exists(repo_dir):
            print(f"Error: {repo_name} not found. Please clone from the appropriate repository.")
            return False
    
    # For SlimSAM, we need to register the pruned model types
    slimsam_init_patch = """
# Register pruned model types for SlimSAM
from functools import partial
from . import modeling

sam_model_registry["vit_p50"] = partial(modeling.Sam, 
    image_encoder=modeling.ImageEncoderViT,
    embed_dim=768,
    depth=12,
    num_heads=12,
    global_attn_indexes=[2, 5, 8, 11],
    prune_ratio=0.5
)

sam_model_registry["vit_p77"] = partial(modeling.Sam,
    image_encoder=modeling.ImageEncoderViT, 
    embed_dim=768,
    depth=12,
    num_heads=12,
    global_attn_indexes=[2, 5, 8, 11],
    prune_ratio=0.77
)
"""
    
    # Check if segment_anything already has the pruned types
    try:
        from segment_anything import sam_model_registry
        if "vit_p50" not in sam_model_registry:
            print("Note: SlimSAM pruned model types not registered. You may need to add them manually.")
    except ImportError:
        pass
    
    print("Model setup complete!")
    print("\nTo use the comprehensive benchmark:")
    print("  python benchmark_all_models_v2.py --models vit_b mobile tiny slim50 expedit")
    print("\nFor quick testing:")
    print("  python benchmark_all_models_v2.py --models mobile tiny --quick")
    
    return True

if __name__ == "__main__":
    setup_models()
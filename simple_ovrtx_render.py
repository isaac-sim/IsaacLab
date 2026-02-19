#!/usr/bin/env python3
"""Simple OVRTX rendering example.

This script demonstrates how to use OVRTX to:
1. Load a USD scene file
2. Render a specific render product
3. Save the output as an image to disk
"""

import os
import sys
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

# Set environment variables for OVRTX
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

# Set LD_PRELOAD if needed (libcarb.so)
libcarb_path = Path.home() / "dev/kit.1/kit/_build/linux-x86_64/release/libcarb.so"
if libcarb_path.exists() and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = str(libcarb_path)

# Add ovrtx Python bindings to path
ovrtx_bindings_path = Path("/home/ncournia/dev/kit.1/rendering/ovrtx/public/bindings/python")
if str(ovrtx_bindings_path) not in sys.path:
    sys.path.insert(0, str(ovrtx_bindings_path))

# Set library path hint before importing ovrtx
import ovrtx._src.bindings as bindings
bindings.OVRTX_LIBRARY_PATH_HINT = "/home/ncournia/dev/kit.1/rendering/_build/linux-x86_64/release"

from ovrtx import Renderer, RendererConfig


def main():
    """Main function to demonstrate OVRTX rendering."""
    
    # Configuration
    usd_scene_path = "/home/ncournia/dev/kit.0/rendering/data/usd/tests/ovrtx/simple_scene.usda"
    # usd_scene_path = "two_envs_dark.usda"
    render_product_paths = [
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        # "/Render/OmniverseKit/HydraTextures/ViewportTexture1"  # Try one at a time first
        #"/Render/RenderProduct_0",
        #"/Render/RenderProduct_1"
    ]
    output_image_paths = [
        "ovrtx_output_viewport0_rgb.png",
        # "ovrtx_output_viewport1_rgb.png"
    ]
    output_depth_paths = [
        "ovrtx_output_viewport0_depth.png",
        # "ovrtx_output_viewport1_depth.png"
    ]
    
    print("="*80)
    print("Simple OVRTX Rendering Example")
    print("="*80)
    print(f"USD Scene: {usd_scene_path}")
    print(f"Render Products: {render_product_paths}")
    print(f"Output RGB Images: {output_image_paths}")
    print(f"Output Depth Images: {output_depth_paths}")
    print()
    
    # Initialize warp
    wp.init()
    print("✓ Warp initialized")
    
    # Create renderer with configuration
    print("\nCreating OVRTX renderer...")
    renderer_config = RendererConfig()
    renderer = Renderer(renderer_config)
    print("✓ OVRTX renderer created")
    
    # Load USD scene
    print(f"\nLoading USD scene: {usd_scene_path}")
    if not Path(usd_scene_path).exists():
        print(f"ERROR: USD file not found: {usd_scene_path}")
        return 1
    
    usd_handle = renderer.add_usd(usd_scene_path, path_prefix=None)
    print(f"✓ USD scene loaded (handle: {usd_handle})")
    
    # Render the scene with both render products
    print(f"\nRendering products: {render_product_paths}")
    try:
        # Verify render products exist in the USD
        print("Note: This script assumes the render products are properly defined in the USD file")
        print("If rendering fails, check that the USD file contains these render products")
        
        # Step the renderer to produce frames for both render products
        products = renderer.step(
            render_products=set(render_product_paths),
            delta_time=1.0/60.0  # 60 FPS
        )
        
        print("✓ Rendering complete")
        print(f"  Products returned: {list(products.keys())}")
        
        # Process each render product
        for idx, render_product_path in enumerate(render_product_paths):
            output_image_path = output_image_paths[idx]
            
            print(f"\nProcessing render product {idx}: {render_product_path}")
            
            if render_product_path not in products:
                print(f"  WARNING: Render product not found in output, skipping")
                continue
            
            # Extract the rendered frame
            product = products[render_product_path]
            print(f"  Number of frames: {len(product.frames)}")
            
            if len(product.frames) == 0:
                print(f"  WARNING: No frames produced for this product, skipping")
                continue
            
            frame = product.frames[0]
            print(f"  Available render vars: {list(frame.render_vars.keys())}")
            
            # Extract the color data (LdrColor is typical for color output)
            if "LdrColor" in frame.render_vars:
                render_var = frame.render_vars["LdrColor"]
                print(f"  Got render var: {render_var}")
                
                # Map the data to access it
                with render_var.map(device="cuda") as mapping:
                    # Convert to warp array
                    rendered_data_wp = wp.from_dlpack(mapping.tensor)
                    
                    # Get shape information
                    shape = rendered_data_wp.shape
                    print(f"  RGB Image shape: {shape}")
                    
                    # Convert to numpy for saving
                    rendered_data_torch = wp.to_torch(rendered_data_wp)
                    rendered_data_np = rendered_data_torch.cpu().numpy()
                    
                    # Convert from float [0, 1] to uint8 [0, 255] if needed
                    if rendered_data_np.dtype in [np.float32, np.float64]:
                        rendered_data_np = (rendered_data_np * 255).astype(np.uint8)
                    
                    # Save as image
                    if len(rendered_data_np.shape) == 3:
                        if rendered_data_np.shape[2] == 4:
                            # RGBA image
                            image = Image.fromarray(rendered_data_np, mode='RGBA')
                        elif rendered_data_np.shape[2] == 3:
                            # RGB image
                            image = Image.fromarray(rendered_data_np, mode='RGB')
                        else:
                            print(f"  WARNING: Unexpected number of channels: {rendered_data_np.shape[2]}, skipping")
                            continue
                    elif len(rendered_data_np.shape) == 2:
                        # Grayscale image
                        image = Image.fromarray(rendered_data_np, mode='L')
                    else:
                        print(f"  WARNING: Unexpected image shape: {rendered_data_np.shape}, skipping")
                        continue
                    
                    image.save(output_image_path)
                    print(f"  ✓ RGB Image saved: {output_image_path}")
                    print(f"    Resolution: {image.width}x{image.height}")
                    print(f"    Mode: {image.mode}")
            else:
                print(f"  WARNING: 'LdrColor' render var not found, skipping RGB")
            
            # Extract depth data if available
            depth_var_names = ["Depth", "depth", "DistanceToImagePlane", "DistanceToImagePlaneSD", "linearDepth", "DepthSD"]
            depth_var_found = None
            for depth_name in depth_var_names:
                if depth_name in frame.render_vars:
                    depth_var_found = depth_name
                    break
            
            if depth_var_found:
                print(f"  Found depth render var: {depth_var_found}")
                render_var = frame.render_vars[depth_var_found]
                print(f"  Got render var: {render_var}")
                
                # Debug: Print handle before mapping
                print(f"  Depth handle: {hex(render_var.handle)} = {render_var.handle}")
                print(f"  Depth handle upper 32 bits (frame?): {render_var.handle >> 32}")
                print(f"  Depth handle lower 32 bits (aov?): {render_var.handle & 0xFFFFFFFF}")
                
                # Also check LdrColor handle for comparison
                if "LdrColor" in frame.render_vars:
                    ldr_handle = frame.render_vars["LdrColor"].handle
                    print(f"  LdrColor handle: {hex(ldr_handle)} = {ldr_handle}")
                    print(f"  LdrColor handle upper 32 bits: {ldr_handle >> 32}")
                    print(f"  LdrColor handle lower 32 bits: {ldr_handle & 0xFFFFFFFF}")
                
                try:
                    # Map the depth data
                    print(f"  Attempting to map depth with handle {hex(render_var.handle)}...")
                    with render_var.map(device="cuda") as mapping:
                        print(f"  mapped render var: {render_var}")
                        # Convert to warp array
                        depth_data_wp = wp.from_dlpack(mapping.tensor)
                        
                        # Get shape information
                        shape = depth_data_wp.shape
                        print(f"  Depth data shape: {shape}")
                        
                        # Convert to numpy for saving
                        depth_data_torch = wp.to_torch(depth_data_wp)
                        depth_data_np = depth_data_torch.cpu().numpy()
                        
                        print(f"  Depth data dtype: {depth_data_np.dtype}")
                        print(f"  Depth range: [{depth_data_np.min():.3f}, {depth_data_np.max():.3f}]")
                        
                        # Normalize depth for visualization
                        # Handle single channel depth
                        if len(depth_data_np.shape) == 3 and depth_data_np.shape[2] == 1:
                            depth_data_np = depth_data_np[:, :, 0]
                        
                        # Normalize to [0, 255] for visualization
                        depth_min = depth_data_np.min()
                        depth_max = depth_data_np.max()
                        if depth_max > depth_min:
                            depth_normalized = (depth_data_np - depth_min) / (depth_max - depth_min)
                            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                        else:
                            depth_uint8 = np.zeros_like(depth_data_np, dtype=np.uint8)
                        
                        # Save as grayscale image
                        depth_image = Image.fromarray(depth_uint8, mode='L')
                        output_depth_path = output_depth_paths[idx]
                        depth_image.save(output_depth_path)
                        print(f"  ✓ Depth image saved: {output_depth_path}")
                        print(f"    Resolution: {depth_image.width}x{depth_image.height}")
                        print(f"    Mode: {depth_image.mode}")
                except Exception as e:
                    print(f"  ERROR extracting depth: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  WARNING: No depth render var found (tried: {depth_var_names})")
        
    except Exception as e:
        print(f"ERROR during rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Cleanup
    print("\nCleaning up...")
    renderer.remove_usd(usd_handle)
    print("✓ Done!")
    
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())

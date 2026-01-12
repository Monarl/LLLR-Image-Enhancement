import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import MambaIR's official metrics
import sys
sys.path.insert(0, '/workspace')
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

def imread(img_path):
    """Read image and convert BGR to RGB"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def adjust_brightness(img, factor=2.0):
    """Adjust image brightness for better visibility"""
    img_adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img_adjusted

def calculate_metrics(img1, img2, crop_border=4, test_y_channel=False):
    """Calculate PSNR and SSIM using MambaIR's official implementation"""
    # Ensure images are the same size
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        return None, None
    
    # Use MambaIR's official metrics with border cropping
    psnr_value = calculate_psnr(img1, img2, crop_border=crop_border, 
                                 input_order='HWC', test_y_channel=test_y_channel)
    ssim_value = calculate_ssim(img1, img2, crop_border=crop_border,
                                input_order='HWC', test_y_channel=test_y_channel)
    
    return psnr_value, ssim_value

def find_ground_truth(original_name, gt_folder):
    """Find ground truth image with flexible name matching"""
    # Try exact match first
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        exact_match = os.path.join(gt_folder, original_name + ext)
        if os.path.exists(exact_match):
            return exact_match
    
    # Try removing common suffixes (like -2.5, x4, etc.)
    # Extract base name by removing numeric suffixes
    base_name = original_name.split('-')[0].split('x')[0]
    
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        base_match = os.path.join(gt_folder, base_name + ext)
        if os.path.exists(base_match):
            print(f"  Matched '{original_name}' to ground truth '{base_name}{ext}'")
            return base_match
    
    return None

def visualize_sr_results(input_folder, output_folder, gt_folder, save_folder, 
                        enhance=False, brightness_factor=2.5, use_metrics=False,
                        crop_border=4, test_y_channel=False):
    """Visualize low-resolution and super-resolved images side by side with metrics"""
    
    # Create save folder
    os.makedirs(save_folder, exist_ok=True)
    
    # Get all output images
    output_list = sorted(glob.glob(os.path.join(output_folder, '*_test_single_image.png')))
    
    if not output_list:
        print(f"No output images found in {output_folder}")
        return
    
    print(f"Found {len(output_list)} output images to visualize")
    print(f"Brightness enhancement: {'ON (factor=' + str(brightness_factor) + ')' if enhance else 'OFF'}")
    print(f"Metrics calculation: {'ON' if use_metrics else 'OFF'}")
    
    results_summary = []
    
    for output_path in output_list:
        basename = os.path.basename(output_path)
        original_name = basename.replace('_test_single_image.png', '')
        
        # Find corresponding input image
        input_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            possible_input = os.path.join(input_folder, original_name + ext)
            if os.path.exists(possible_input):
                input_path = possible_input
                break
        
        if input_path is None:
            print(f"No input found for output: {basename}")
            continue
        
        # Find ground truth if metrics are enabled
        gt_path = None
        img_gt = None
        if use_metrics and gt_folder:
            gt_path = find_ground_truth(original_name, gt_folder)
            if gt_path:
                img_gt = imread(gt_path)
            else:
                print(f"  Warning: No ground truth found for '{original_name}'")
        
        # Read images
        img_input = imread(input_path)
        img_output = imread(output_path)
        
        if img_input is None or img_output is None:
            continue
        
        # Calculate metrics if ground truth is available
        psnr_bicubic, ssim_bicubic = None, None
        psnr_sr, ssim_sr = None, None
        
        if use_metrics and img_gt is not None:
            # Upscale input with bicubic for fair comparison
            img_input_upscaled = cv2.resize(img_input, (img_output.shape[1], img_output.shape[0]), 
                                           interpolation=cv2.INTER_CUBIC)
            
            # Calculate metrics using MambaIR's official implementation
            psnr_bicubic, ssim_bicubic = calculate_metrics(img_gt, img_input_upscaled, 
                                                          crop_border=crop_border,
                                                          test_y_channel=test_y_channel)
            
            psnr_sr, ssim_sr = calculate_metrics(img_gt, img_output,
                                                crop_border=crop_border,
                                                test_y_channel=test_y_channel)
            
            # Store results
            if psnr_sr is not None and psnr_bicubic is not None:
                results_summary.append({
                    'image': original_name,
                    'bicubic_psnr': psnr_bicubic,
                    'bicubic_ssim': ssim_bicubic,
                    'sr_psnr': psnr_sr,
                    'sr_ssim': ssim_sr,
                    'psnr_gain': psnr_sr - psnr_bicubic,
                    'ssim_gain': ssim_sr - ssim_bicubic
                })
        
        # Apply brightness adjustment if enabled
        if enhance:
            img_input_display = adjust_brightness(img_input, factor=brightness_factor)
            img_output_display = adjust_brightness(img_output, factor=brightness_factor)
            title_suffix = f' (Enhanced {brightness_factor}x)'
        else:
            img_input_display = img_input
            img_output_display = img_output
            title_suffix = ''
        
        # Create comparison figure
        num_cols = 3 if (use_metrics and img_gt is not None) else 2
        fig = plt.figure(figsize=(8 * num_cols, 7))
        
        ax1 = fig.add_subplot(1, num_cols, 1)
        title1 = f'LR{title_suffix}\n{img_input.shape[1]}x{img_input.shape[0]}'
        if psnr_bicubic is not None:
            title1 += f'\nBicubic: PSNR={psnr_bicubic:.2f}dB, SSIM={ssim_bicubic:.4f}'
        plt.title(title1, fontsize=14)
        ax1.axis('off')
        ax1.imshow(img_input_display)
        
        ax2 = fig.add_subplot(1, num_cols, 2)
        title2 = f'SR (MambaIR){title_suffix}\n{img_output.shape[1]}x{img_output.shape[0]}'
        if psnr_sr is not None:
            title2 += f'\nPSNR={psnr_sr:.2f}dB, SSIM={ssim_sr:.4f}'
            title2 += f'\n(+{psnr_sr - psnr_bicubic:.2f}dB, +{ssim_sr - ssim_bicubic:.4f})'
        plt.title(title2, fontsize=14, color='green' if psnr_sr and psnr_sr > psnr_bicubic else 'black')
        ax2.axis('off')
        ax2.imshow(img_output_display)
        
        # Add ground truth if available
        if use_metrics and img_gt is not None:
            ax3 = fig.add_subplot(1, num_cols, 3)
            img_gt_display = adjust_brightness(img_gt, factor=brightness_factor) if enhance else img_gt
            plt.title(f'Ground Truth (HR){title_suffix}\n{img_gt.shape[1]}x{img_gt.shape[0]}', fontsize=14)
            ax3.axis('off')
            ax3.imshow(img_gt_display)
        
        plt.tight_layout()
        
        # Save comparison
        suffix = f'_enhanced_{brightness_factor}x' if enhance else ''
        suffix += '_metrics' if use_metrics else ''
        save_path = os.path.join(save_folder, f"{original_name}_comparison{suffix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create zoomed-in crop comparison
        h_in, w_in = img_input.shape[:2]
        h_out, w_out = img_output.shape[:2]
        
        crop_h_in = h_in // 4
        crop_w_in = w_in // 4
        crop_h_out = h_out // 4
        crop_w_out = w_out // 4
        
        center_y_in, center_x_in = h_in // 2, w_in // 2
        center_y_out, center_x_out = h_out // 2, w_out // 2
        
        crop_input = img_input[
            center_y_in - crop_h_in:center_y_in + crop_h_in,
            center_x_in - crop_w_in:center_x_in + crop_w_in
        ]
        crop_output = img_output[
            center_y_out - crop_h_out:center_y_out + crop_h_out,
            center_x_out - crop_w_out:center_x_out + crop_w_out
        ]
        
        # Resize input crop to match output size
        crop_input_resized = cv2.resize(crop_input, (crop_output.shape[1], crop_output.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
        
        # Crop ground truth if available
        crop_gt = None
        if img_gt is not None:
            crop_gt = img_gt[
                center_y_out - crop_h_out:center_y_out + crop_h_out,
                center_x_out - crop_w_out:center_x_out + crop_w_out
            ]
        
        # Apply brightness if enabled
        if enhance:
            crop_input_display = adjust_brightness(crop_input_resized, factor=brightness_factor)
            crop_output_display = adjust_brightness(crop_output, factor=brightness_factor)
            crop_gt_display = adjust_brightness(crop_gt, factor=brightness_factor) if crop_gt is not None else None
        else:
            crop_input_display = crop_input_resized
            crop_output_display = crop_output
            crop_gt_display = crop_gt
        
        # Create crop comparison
        num_crop_cols = 3 if crop_gt is not None else 2
        fig2 = plt.figure(figsize=(8 * num_crop_cols, 8))
        
        ax1 = fig2.add_subplot(1, num_crop_cols, 1)
        plt.title(f'LR (Bicubic upscaled){title_suffix} - Center Crop', fontsize=16)
        ax1.axis('off')
        ax1.imshow(crop_input_display)
        
        ax2 = fig2.add_subplot(1, num_crop_cols, 2)
        plt.title(f'SR (MambaIR){title_suffix} - Center Crop', fontsize=16)
        ax2.axis('off')
        ax2.imshow(crop_output_display)
        
        if crop_gt_display is not None:
            ax3 = fig2.add_subplot(1, num_crop_cols, 3)
            plt.title(f'Ground Truth{title_suffix} - Center Crop', fontsize=16)
            ax3.axis('off')
            ax3.imshow(crop_gt_display)
        
        plt.tight_layout()
        
        crop_save_path = os.path.join(save_folder, f"{original_name}_crop_comparison{suffix}.png")
        plt.savefig(crop_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Processed: {original_name}")
        print(f"  Input: {os.path.basename(input_path)} - Size: {img_input.shape}")
        print(f"  Output: {basename} - Size: {img_output.shape}")
        if gt_path:
            print(f"  Ground Truth: {os.path.basename(gt_path)} - Size: {img_gt.shape}")
        if psnr_bicubic is not None:
            print(f"  Bicubic - PSNR: {psnr_bicubic:.2f}dB, SSIM: {ssim_bicubic:.4f}")
        if psnr_sr is not None:
            print(f"  MambaIR - PSNR: {psnr_sr:.2f}dB, SSIM: {ssim_sr:.4f}")
            print(f"  Improvement: +{psnr_sr - psnr_bicubic:.2f}dB, +{ssim_sr - ssim_bicubic:.4f}")
        print(f"  Saved full comparison: {save_path}")
        print(f"  Saved crop comparison: {crop_save_path}")
        print("-" * 70)
    
    # Print summary
    if results_summary:
        print("\n" + "=" * 70)
        print("METRICS SUMMARY")
        print("=" * 70)
        print(f"{'Image':<20} {'Bicubic PSNR':<15} {'SR PSNR':<15} {'Gain':<10}")
        print(f"{'     ':<20} {'Bicubic SSIM':<15} {'SR SSIM':<15} {'Gain':<10}")
        print("-" * 70)
        
        for result in results_summary:
            print(f"{result['image']:<20} "
                  f"{result['bicubic_psnr']:>8.2f}dB      "
                  f"{result['sr_psnr']:>8.2f}dB      "
                  f"{result['psnr_gain']:>+7.2f}dB")
            print(f"{'     ':<20} "
                  f"{result['bicubic_ssim']:>8.4f}       "
                  f"{result['sr_ssim']:>8.4f}       "
                  f"{result['ssim_gain']:>+7.4f}")
            print()
        
        # Calculate averages
        avg_psnr_gain = np.mean([r['psnr_gain'] for r in results_summary])
        avg_ssim_gain = np.mean([r['ssim_gain'] for r in results_summary])
        
        print("-" * 70)
        print(f"Average Improvement: PSNR +{avg_psnr_gain:.2f}dB, SSIM +{avg_ssim_gain:.4f}")
        print("=" * 70)
    elif use_metrics:
        print("\nNo ground truth images found - metrics cannot be calculated.")
        print("Make sure ground truth images exist in:", gt_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Super-Resolution results')
    parser.add_argument('--enhance', action='store_true', 
                        help='Enable brightness enhancement')
    parser.add_argument('--brightness', type=float, default=2.5,
                        help='Brightness factor (default: 2.5)')
    parser.add_argument('--metrics', action='store_true',
                        help='Calculate PSNR and SSIM metrics (requires ground truth)')
    parser.add_argument('--crop-border', type=int, default=4,
                        help='Crop border pixels for metrics (default: 4)')
    parser.add_argument('--test-y-channel', action='store_true',
                        help='Test on Y channel only (like official benchmarks)')
    parser.add_argument('--input', type=str, default='/workspace/samples/inputs',
                        help='Input folder path')
    parser.add_argument('--output', type=str, default='/workspace/results/test_single_image/visualization/MyImages',
                        help='Output folder path')
    parser.add_argument('--gt', type=str, default='/workspace/samples/results',
                        help='Ground truth folder path (for metrics)')
    parser.add_argument('--save', type=str, default='/workspace/results/test_single_image/comparisons',
                        help='Save folder path')
    
    args = parser.parse_args()
    
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    if args.metrics:
        print(f"Ground truth folder: {args.gt}")
        print(f"Crop border: {args.crop_border} pixels")
        print(f"Test Y-channel only: {args.test_y_channel}")
    print(f"Comparisons will be saved to: {args.save}")
    print("=" * 70)
    
    visualize_sr_results(args.input, args.output, args.gt, args.save, 
                        enhance=args.enhance, brightness_factor=args.brightness,
                        use_metrics=args.metrics, crop_border=args.crop_border,
                        test_y_channel=args.test_y_channel)
    
    print("\n✓ All comparisons saved!")
    print(f"View them in: {args.save}")
    if args.metrics:
        print("\nMetrics show quantitative improvement of MambaIR over bicubic upscaling!")
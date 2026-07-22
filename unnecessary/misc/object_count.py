import numpy as np
import tifffile
import time
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from scipy import ndimage

def parse_config(config_path):
    """
    Parse config_HBM.txt file
    
    Returns:
    --------
    config : dict
        Configuration parameters
    """
    config = {}
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse key = value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert to appropriate type
                if value.lower() == 'true':
                    config[key] = True
                elif value.lower() == 'false':
                    config[key] = False
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    # Numeric value
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                else:
                    config[key] = value
    
    return config


def find_files_in_folder(folder_path):
    """
    Find required files in folder:
    - Bump file (contains "bump3D.tif")
    - Void file (contains "voidsOnly.tif")
    - Config file (contains "config_HBM.txt")
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing files
    
    Returns:
    --------
    files : dict
        Dictionary with keys: 'tgv', 'void', 'config'
    """
    files = {'bump': None, 'void': None, 'config': None}
    
    # Search all files in folder
    all_files = glob.glob(os.path.join(folder_path, '*'))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        if 'bump3D.tif' in filename:
            files['bump'] = file_path
        elif 'voidsOnly.tif' in filename:
            files['void'] = file_path
        elif 'config_HBM.txt' in filename:
            files['config'] = file_path
    
    # Validate all files found
    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing files in folder: {missing}")
    
    return files


def calculate_statistics_optimized(labeled_class1, class2_binary, num_objects, 
                                   start_slice, end_slice):
    """
    ULTRA-OPTIMIZED statistics calculation
    Only process relevant slices (start_slice to end_slice)
    
    Parameters:
    -----------
    labeled_class1 : ndarray
        Labeled 3D stack
    class2_binary : ndarray
        Binary void stack
    num_objects : int
        Number of objects
    start_slice : int
        First slice to process
    end_slice : int
        Last slice to process
    
    Returns:
    --------
    stats : list
        Statistics for each object
    """
    
    # OPTIMIZATION 1: Extract only relevant slices
    labeled_roi = labeled_class1[start_slice:end_slice+1]
    void_roi = class2_binary[start_slice:end_slice+1]
    
    # OPTIMIZATION 2: Vectorized counting on ROI only
    # Count TGV volumes
    unique_labels, tgv_counts = np.unique(labeled_roi[labeled_roi > 0], 
                                           return_counts=True)
    tgv_volume_dict = dict(zip(unique_labels, tgv_counts))
    
    # OPTIMIZATION 3: Multiply arrays once instead of per-object iteration
    void_labeled = labeled_roi * void_roi
    
    # Count void volumes
    if void_labeled.max() > 0:
        unique_void_labels, void_counts = np.unique(void_labeled[void_labeled > 0], 
                                                      return_counts=True)
        void_volume_dict = dict(zip(unique_void_labels, void_counts))
    else:
        void_volume_dict = {}
    
    # OPTIMIZATION 4: Pre-compute all object masks at once using broadcasting
    # This is faster than computing masks one by one
    all_obj_ids = np.arange(1, num_objects + 1)
    
    # OPTIMIZATION 5: Use np.isin for faster multi-label checking
    objects_present = np.isin(all_obj_ids, unique_labels)
    
    stats = []
    
    # OPTIMIZATION 6: Only process objects that actually exist in ROI
    for obj_id in all_obj_ids[objects_present]:
        tgv_vol = tgv_volume_dict.get(obj_id, 0)
        void_vol = void_volume_dict.get(obj_id, 0)
        
        # OPTIMIZATION 7: Fast bbox calculation using np.where on ROI
        obj_mask = (labeled_roi == obj_id)
        obj_indices = np.where(obj_mask)
        
        if len(obj_indices[0]) > 0:
            z_coords, y_coords, x_coords = obj_indices
            bbox = {
                'z_min': int(z_coords.min()) + start_slice,  # Offset back to original
                'z_max': int(z_coords.max()) + start_slice,
                'y_min': int(y_coords.min()),
                'y_max': int(y_coords.max()),
                'x_min': int(x_coords.min()),
                'x_max': int(x_coords.max()),
            }
        else:
            bbox = None
        
        stats.append({
            'object_id': int(obj_id),
            'tgv_volume': int(tgv_vol),
            'void_volume': int(void_vol),
            'ratio': float(void_vol / (tgv_vol + void_vol)) if (tgv_vol + void_vol) > 0 else None,
            'bbox': bbox
        })
    
    return stats


def format_number(num):
    """Format large numbers with K/M suffix"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num}"


def find_best_text_position(obj_mask_slice, width, height, box_width, box_height):
    """
    Find the best position for text box to avoid overlapping with object
    
    Priority:
    1. Right side of object
    2. Left side of object
    3. Top of object
    4. Bottom of object
    5. Overlay (fallback)
    """
    obj_coords = np.where(obj_mask_slice)
    
    if len(obj_coords[1]) == 0:
        return None, None, None
    
    # Get object boundaries
    y_min, y_max = obj_coords[0].min(), obj_coords[0].max()
    x_min, x_max = obj_coords[1].min(), obj_coords[1].max()
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2
    
    PADDING = 8
    
    # 1. Try RIGHT side (preferred)
    text_x_right = x_max + PADDING
    text_y_right = max(PADDING, min(y_center - box_height // 2, height - box_height - PADDING))
    
    if text_x_right + box_width <= width - PADDING:
        return text_x_right, text_y_right, 'right'
    
    # 2. Try LEFT side
    text_x_left = x_min - box_width - PADDING
    text_y_left = max(PADDING, min(y_center - box_height // 2, height - box_height - PADDING))
    
    if text_x_left >= PADDING:
        return text_x_left, text_y_left, 'left'
    
    # 3. Try TOP
    text_y_top = y_min - box_height - PADDING
    text_x_top = max(PADDING, min(x_center - box_width // 2, width - box_width - PADDING))
    
    if text_y_top >= PADDING:
        return text_x_top, text_y_top, 'top'
    
    # 4. Try BOTTOM
    text_y_bottom = y_max + PADDING
    text_x_bottom = max(PADDING, min(x_center - box_width // 2, width - box_width - PADDING))
    
    if text_y_bottom + box_height <= height - PADDING:
        return text_x_bottom, text_y_bottom, 'bottom'
    
    # 5. Fallback: overlay
    return width - box_width - PADDING, PADDING, 'overlay'


def process_folder(folder_path):
    """
    Main processing function
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing TGV, Void, and config files
    """
    print("="*70)
    print("BUMP + VOID MEASUREMENT ANALYSIS")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Find files
    print("\n[Step 1] Finding files in folder...")
    files = find_files_in_folder(folder_path)
    print(f"  ✓ Bump file:    {os.path.basename(files['bump'])}")
    print(f"  ✓ Void file:   {os.path.basename(files['void'])}")
    print(f"  ✓ Config file: {os.path.basename(files['config'])}")
    
    # 2. Parse config
    print("\n[Step 2] Parsing configuration...")
    config = parse_config(files['config'])
    
    top_air_end = config.get('TOP_AIR_END_SLICE_NUM', -1)
    btm_air_start = config.get('BTM_AIR_START_SLICE_NUM', -1)
    
    print(f"  ✓ Measurement range: Slice {top_air_end} to {btm_air_start}")
    print(f"  ✓ Total measurement slices: {btm_air_start - top_air_end + 1}")
    
    # 3. Read TIFF stacks
    print("\n[Step 3] Reading TIFF stacks...")
    read_start = time.time()
    
    bump_stack = tifffile.imread(files['bump'])
    void_stack = tifffile.imread(files['void'])
    
    print(f"  ✓ Reading time: {time.time() - read_start:.2f}s")
    
    # OPTIMIZATION: Convert only ROI to binary
    depth, height, width = bump_stack.shape
    print(f"  ✓ Stack shape: {depth} slices x {height} x {width} pixels")
    
    # Validate slice range
    if top_air_end < 0 or btm_air_start >= depth:
        raise ValueError(f"Invalid slice range: {top_air_end}-{btm_air_start} (stack has {depth} slices)")
    
    print(f"  ✓ Processing only ROI: slices {top_air_end}-{btm_air_start}")
    
    # Convert to binary (full stack needed for labeling continuity)
    convert_start = time.time()
    bump_binary = (bump_stack > 127).astype(np.uint8)
    void_binary = (void_stack > 127).astype(np.uint8)
    print(f"  ✓ Binary conversion time: {time.time() - convert_start:.2f}s")
    
    # Free memory
    del bump_stack, void_stack
    
    # 4. Label 3D objects (need full stack for connectivity)
    print("\n[Step 4] Detecting 3D objects (connected components)...")
    label_start = time.time()
    
    # OPTIMIZATION: Label only ROI + small buffer for connectivity
    buffer = 2
    roi_start = max(0, top_air_end - buffer)
    roi_end = min(depth, btm_air_start + buffer + 1)
    
    print(f"  ✓ Labeling ROI: slices {roi_start}-{roi_end} (with {buffer}-slice buffer)")
    
    labeled_bump_roi, num_objects = ndimage.label(bump_binary[roi_start:roi_end])
    
    # Create full labeled stack (save memory by only storing ROI)
    labeled_bump = np.zeros_like(bump_binary, dtype=np.uint16)
    labeled_bump[roi_start:roi_end] = labeled_bump_roi
    
    del labeled_bump_roi
    
    print(f"  ✓ Found {num_objects} Bump object(s)")
    print(f"  ✓ Labeling time: {time.time() - label_start:.2f}s")
    
    # 5. Calculate statistics (OPTIMIZED - only on measurement range)
    print("\n[Step 5] Calculating statistics (optimized)...")
    stats_start = time.time()
    
    stats = calculate_statistics_optimized(
        labeled_bump, 
        void_binary, 
        num_objects,
        top_air_end,
        btm_air_start
    )
    
    print(f"  ✓ Statistics calculated in {time.time() - stats_start:.2f}s")
    print(f"  ✓ Processed only {btm_air_start - top_air_end + 1}/{depth} slices")
    
    # Print summary
    print("\n" + "-"*70)
    print("MEASUREMENT SUMMARY")
    print("-"*70)
    
    for obj_stat in stats:
        print(f"\nObject #{obj_stat['object_id']}:")
        print(f"  Bump volume:   {obj_stat['tgv_volume']:>12,} voxels")
        print(f"  Void volume:  {obj_stat['void_volume']:>12,} voxels")
        
        if obj_stat['tgv_volume'] > 0:
            print(f"  Ratio:        {obj_stat['ratio']*100:>11.3f}%")
        else:
            print(f"  Ratio:        {'N/A':>12}")
    
    print("-"*70)
    print(f"Total TGV volume:   {sum(s['tgv_volume'] for s in stats):>12,} voxels")
    print(f"Total Void volume:  {sum(s['void_volume'] for s in stats):>12,} voxels")
    print("-"*70)
    
    # 6. Export CSV
    print("\n[Step 6] Exporting statistics to CSV...")
    csv_path = os.path.join(folder_path, "measurement_statistics.csv")
    export_to_csv(stats, csv_path)
    
    # 7. Create annotated visualization
    print("\n[Step 7] Creating annotated visualization...")
    output_path = os.path.join(folder_path, "annotated_measurement.tif")
    
    create_annotated_stack_optimized(
        labeled_tgv,
        void_binary,
        stats,
        top_air_end+10,
        btm_air_start,
        output_path
    )
    
    # 8. Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    print(f"\nOutput files:")
    print(f"  1. {csv_path}")
    print(f"  2. {output_path}")
    print("="*70)


def create_annotated_stack_optimized(labeled_tgv, void_binary, stats, 
                                     start_slice, end_slice, output_path):
    """
    Create annotated stack with text positions calculated ONCE from start_slice
    and copied to all subsequent slices
    
    OPTIMIZATION: Only create annotated slices in measurement range
    """
    print(f"  Processing slices {start_slice} to {end_slice}...")
    
    depth, height, width = labeled_bump.shape
    
    # OPTIMIZATION: Only create RGB stack for measurement range
    num_slices = end_slice - start_slice + 1
    annotated_stack = np.zeros((num_slices, height, width, 3), dtype=np.uint8)
    
    # Box dimensions
    BOX_WIDTH = 55
    BOX_HEIGHT = 70
    
    # Load fonts
    try:
        font_large_size = 16
        font_xlarge_size = 20
        
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
        font_large = None
        font_xlarge = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_large = ImageFont.truetype(font_path, font_large_size)
                font_xlarge = ImageFont.truetype(font_path, font_xlarge_size)
                break
        
        if font_large is None:
            font_large = ImageFont.load_default()
            font_xlarge = ImageFont.load_default()
    except:
        font_large = ImageFont.load_default()
        font_xlarge = ImageFont.load_default()
    
    # STEP 1: Calculate text positions from reference slice (start_slice)
    print(f"  Calculating text positions from slice {start_slice}...")
    
    text_positions = {}  # {obj_id: (id_x, id_y, data_x, data_y, pos_type)}
    
    ref_slice = labeled_bump[start_slice]
    
    for obj_stat in stats:
        obj_id = obj_stat['object_id']
        bbox = obj_stat['bbox']
        
        if bbox is None:
            continue
        
        # Check if object exists in reference slice
        if bbox['z_min'] <= start_slice <= bbox['z_max']:
            obj_mask_slice = (ref_slice == obj_id)
            
            if np.any(obj_mask_slice):
                obj_coords = np.where(obj_mask_slice)
                
                # Calculate ID position (top of object)
                y_min = obj_coords[0].min()
                x_min = obj_coords[1].min()
                x_max = obj_coords[1].max()
                x_center = (x_min + x_max) // 2
                
                id_x = x_center
                id_y = max(5, y_min - 25)
                
                # Calculate data box position
                data_x, data_y, pos_type = find_best_text_position(
                    obj_mask_slice, width, height, BOX_WIDTH, BOX_HEIGHT
                )
                
                if data_x is not None:
                    text_positions[obj_id] = {
                        'id_x': id_x,
                        'id_y': id_y,
                        'data_x': data_x,
                        'data_y': data_y,
                        'pos_type': pos_type
                    }
    
    print(f"  ✓ Calculated positions for {len(text_positions)} objects")
    
    # STEP 2: Apply same positions to all slices in range
    print(f"  Annotating {num_slices} slices...")
    
    annotation_start = time.time()
    
    for z in range(start_slice, end_slice + 1):
        z_idx = z - start_slice  # Index in output stack
        
        if z_idx % 50 == 0:
            elapsed = time.time() - annotation_start
            progress = (z_idx / num_slices) * 100
            estimated_total = elapsed / (z_idx + 1) * num_slices if z_idx > 0 else 0
            remaining = estimated_total - elapsed
            print(f"    Progress: {progress:.1f}% ({z_idx}/{num_slices}) - "
                  f"Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s", end='\r')
        
        # Create base visualization
        slice_img = annotated_stack[z_idx]
        
        # Bump in red
        bump_mask = labeled_bump[z] > 0
        slice_img[bump_mask, 0] = 255
        
        # Void in cyan
        void_mask = void_binary[z] > 0
        slice_img[void_mask, 1] = 255
        slice_img[void_mask, 2] = 255
        
        # Convert to PIL
        pil_img = Image.fromarray(slice_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw annotations using FIXED positions
        for obj_stat in stats:
            obj_id = obj_stat['object_id']
            bbox = obj_stat['bbox']
            
            if obj_id not in text_positions:
                continue
            
            # Check if object exists in this slice
            if bbox['z_min'] <= z <= bbox['z_max']:
                obj_mask_slice = (labeled_bump[z] == obj_id)
                
                if np.any(obj_mask_slice):
                    pos = text_positions[obj_id]
                    
                    # Draw Object ID
                    id_text = f"#{obj_id}"
                    
                    try:
                        id_width = draw.textlength(id_text, font=font_xlarge)
                    except:
                        id_width = len(id_text) * 12
                    
                    id_x = pos['id_x'] - id_width // 2
                    id_y = pos['id_y']
                    
                    # ID background
                    draw.rectangle([id_x - 3, id_y - 2, id_x + id_width + 3, id_y + 22],
                                 fill=(0, 0, 0), outline=(255, 255, 0), width=2)
                    draw.text((id_x, id_y), id_text, fill=(255, 255, 0), font=font_xlarge)
                    
                    # Draw data box
                    data_x = pos['data_x']
                    data_y = pos['data_y']
                    pos_type = pos['pos_type']
                    
                    tgv_vol = obj_stat['tgv_volume']
                    void_vol = obj_stat['void_volume']
                    ratio = obj_stat['ratio']
                    
                    data_lines = [
                        format_number(tgv_vol),
                        format_number(void_vol),
                        f"{ratio*100:.3f}%" if ratio else "N/A"
                    ]
                    
                    # Data box background
                    rect_x1 = int(data_x)
                    rect_y1 = int(data_y)
                    rect_x2 = int(data_x + BOX_WIDTH)
                    rect_y2 = int(data_y + BOX_HEIGHT)
                    
                    border_color = (255, 100, 100) if pos_type == 'overlay' else (255, 255, 0)
                    
                    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2],
                                 fill=(0, 0, 0), outline=border_color, width=2)
                    
                    # Draw connector line
                    if pos_type in ['left', 'top', 'bottom']:
                        obj_coords = np.where(obj_mask_slice)
                        obj_y_center = int(np.mean(obj_coords[0]))
                        obj_x_center = int(np.mean(obj_coords[1]))
                        
                        box_center_x = rect_x1 + BOX_WIDTH // 2
                        box_center_y = rect_y1 + BOX_HEIGHT // 2
                        
                        draw.line([(box_center_x, box_center_y), (obj_x_center, obj_y_center)],
                                fill=(255, 255, 0), width=1)
                    
                    # Draw numbers
                    line_spacing = 22
                    current_y = data_y + 8
                    
                    for number in data_lines:
                        try:
                            num_width = draw.textlength(number, font=font_large)
                        except:
                            num_width = len(number) * 10
                        
                        num_x = data_x + (BOX_WIDTH - num_width) // 2
                        draw.text((num_x, current_y), number, fill=(255, 255, 255), font=font_large)
                        current_y += line_spacing
        
        # Convert back to numpy
        annotated_stack[z_idx] = np.array(pil_img)
    
    print(f"\n  ✓ Annotation complete in {time.time() - annotation_start:.2f}s!")
    
    # Save
    print(f"  Saving to {output_path}...")
    save_start = time.time()
    tifffile.imwrite(output_path, annotated_stack, compression='zlib')
    print(f"  ✓ Saved in {time.time() - save_start:.2f}s!")


def export_to_csv(stats, output_csv):
    """Export statistics to CSV file"""
    import csv
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Object #', 'Bump Volume (voxels)', 'Void Volume (voxels)', 'Ratio Void/(Bump+Void)'])
        
        for s in stats:
            ratio_str = f"{s['ratio']*100:.3f}%" if s['ratio'] is not None else "N/A"
            writer.writerow([s['object_id'], s['tgv_volume'], s['void_volume'], ratio_str])
    
    print(f"  ✓ CSV exported: {output_csv}")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    import sys
    
    # Get folder path from command line or use default
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default folder for testing
        folder_path = "E:/semiconductor/DEMO/images/20260213/afternoon/test_online/SG1_6_slice/noooooooo"
    
    # Process folder
    process_folder(folder_path)
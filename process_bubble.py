import cv2
import numpy as np


def get_dominant_color(image, mask=None):
    """
    Get the dominant color of an image region using K-means clustering.
    
    Args:
        image: Input image (BGR)
        mask: Optional mask to specify region of interest
        
    Returns: 
        tuple: Dominant color as (B, G, R)
    """
    if mask is not None:
        # Only get pixels within the mask
        pixels = image[mask == 255]
    else:
        pixels = image.reshape(-1, 3)
    
    if len(pixels) == 0:
        return (255, 255, 255)  # Default white
    
    # Use K-means to find dominant color
    pixels = np.float32(pixels)
    
    # Find 3 main colors, take the most frequent one
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = min(3, len(pixels))  # Ensure k is not greater than number of pixels
    
    if k < 1:
        return (255, 255, 255)
    
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixels for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_color = centers[dominant_idx]
    
    return tuple(int(c) for c in dominant_color)


def get_color_by_histogram(pixels, bins=32):
    """
    Find dominant color using histogram binning.
    More accurate than simple mean/median for multimodal distributions.
    
    Args:
        pixels: Array of pixel colors (N, 3)
        bins: Number of bins per channel
        
    Returns:
        tuple: Dominant color as (B, G, R)
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    # Create 3D histogram
    hist_b = np.histogram(pixels[:, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(pixels[:, 1], bins=bins, range=(0, 256))[0]
    hist_r = np.histogram(pixels[:, 2], bins=bins, range=(0, 256))[0]
    
    # Find peak bin for each channel
    bin_width = 256 // bins
    b_peak = np.argmax(hist_b) * bin_width + bin_width // 2
    g_peak = np.argmax(hist_g) * bin_width + bin_width // 2
    r_peak = np.argmax(hist_r) * bin_width + bin_width // 2
    
    return (int(b_peak), int(g_peak), int(r_peak))


def get_color_by_mode(pixels):
    """
    Find the most frequent color (mode) in pixel array.
    Uses color quantization to group similar colors.
    
    Args:
        pixels: Array of pixel colors (N, 3)
        
    Returns:
        tuple: Most frequent color as (B, G, R)
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    # Quantize colors (reduce to 32 levels per channel)
    quantized = (pixels // 8) * 8
    
    # Convert to hashable format for counting
    color_codes = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    
    # Find most frequent
    unique, counts = np.unique(color_codes, return_counts=True)
    most_freq_code = unique[np.argmax(counts)]
    
    # Decode back to BGR
    b = (most_freq_code // 65536) % 256
    g = (most_freq_code // 256) % 256
    r = most_freq_code % 256
    
    return (int(b), int(g), int(r))


def get_bubble_background_color(image, sample_border=True):
    """
    Detect speech bubble background color by analyzing center region.
    Since bounding box is larger than actual bubble, we sample from center
    (inside the bubble) and filter out text pixels.
    
    Args:
        image: Input bubble image (BGR)
        sample_border: Legacy param, ignored - always samples from center now
        
    Returns: 
        tuple: Background color as (B, G, R)
    """
    h, w = image.shape[:2]
    
    # ===== Sample from CENTER region (inside the bubble) =====
    # Avoid edges since bbox is larger than actual bubble
    margin_y = max(10, h // 5)  # 20% margin from top/bottom
    margin_x = max(10, w // 5)  # 20% margin from left/right
    
    # Get center region
    center_region = image[margin_y:h-margin_y, margin_x:w-margin_x]
    
    if center_region.size == 0:
        # Fallback if image too small
        center_region = image
    
    center_pixels = center_region.reshape(-1, 3)
    
    # ===== Filter out text pixels =====
    # Text is usually very dark (black) or very bright (white)
    # Keep only mid-range pixels that are likely background
    gray_values = np.mean(center_pixels, axis=1)
    
    # Check if this looks like a dark bubble or light bubble
    median_gray = np.median(gray_values)
    
    if median_gray > 128:
        # Light bubble (white/light bg, dark text)
        # Keep pixels that are bright (background), remove dark (text)
        bg_mask = gray_values > 180
    else:
        # Dark bubble (dark bg, white/light text)
        # Keep pixels that are dark (background), remove bright (text)
        bg_mask = gray_values < 80
    
    # Apply mask to get background pixels only
    if np.sum(bg_mask) > 100:
        bg_pixels = center_pixels[bg_mask]
    else:
        # Not enough pixels after filtering, use all center pixels
        bg_pixels = center_pixels
    
    collected_colors = []
    
    # ===== Get colors using different methods =====
    
    # Method 1: Mode from filtered background pixels
    color_mode = get_color_by_mode(bg_pixels)
    collected_colors.append(color_mode)
    
    # Method 4: Histogram from center region
    color_hist = get_color_by_histogram(bg_pixels, bins=32)
    collected_colors.append(color_hist)
    
    # Method: Median (as fallback reference)
    color_median = tuple(int(c) for c in np.median(bg_pixels, axis=0))
    collected_colors.append(color_median)
    
    # ===== Vote for final color =====
    collected_colors = np.array(collected_colors)
    
    # Find color with minimum total distance to others (consensus)
    best_color = collected_colors[0]
    min_total_dist = float('inf')
    
    for i, color in enumerate(collected_colors):
        total_dist = 0
        for j, other_color in enumerate(collected_colors):
            if i != j:
                dist = np.sum(np.abs(color.astype(int) - other_color.astype(int)))
                total_dist += dist
        
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_color = color
    
    return tuple(int(c) for c in best_color)


def is_dark_bubble(image, threshold=100):
    """
    Determine if a bubble image is dark (black bubble with white text).
    
    Args:
        image: Input bubble image (BGR)
        threshold: Intensity threshold (below = dark bubble)
        
    Returns:
        bool: True if dark bubble, False if light bubble
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold


def process_dark_bubble(image, fill_color=None):
    """
    Processes a dark speech bubble (black with white text).
    Fills the bubble contents with the detected or specified color.
    
    Args:
        image (numpy.ndarray): Input dark bubble image.
        fill_color: Color to fill (None = auto-detect)
        
    Returns:
        tuple: (processed_image, largest_contour, fill_color_used)
    """
    if fill_color is None:
        fill_color = get_bubble_background_color(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # For dark bubbles, find the dark region (invert threshold)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = fill_color
        return image, largest_contour, fill_color
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # Fill with detected or specified color
    image[mask == 255] = fill_color
    
    return image, largest_contour, fill_color


def process_bubble(image, fill_color=None):
    """
    Processes the speech bubble in the given image, filling its contents with
    the detected or specified background color. Uses adaptive thresholding
    based on background intensity.

    Parameters:
    - image (numpy.ndarray): Input image.
    - fill_color: Color to fill (None = auto-detect)

    Returns:
    - image (numpy.ndarray): Image with the speech bubble content filled.
    - largest_contour (numpy.ndarray): Contour of the detected speech bubble.
    - fill_color_used (tuple): The color used to fill the bubble.
    """
    if fill_color is None:
        fill_color = get_bubble_background_color(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold based on background color
    bg_intensity = np.mean(fill_color)
    if bg_intensity > 200: 
        # Light background (white)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    elif bg_intensity < 50:
        # Dark background (black)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        # Medium color background - use adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handle case when no contours found
    if not contours:
        # Return original image with a simple rectangular contour
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        # Fill with detected color
        image[:] = fill_color
        return image, largest_contour, fill_color
    
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    image[mask == 255] = fill_color

    return image, largest_contour, fill_color


def process_bubble_auto(image, force_dark=False, custom_color=None):
    """
    Automatically detect bubble type and process accordingly.
    
    Args:
        image: Input bubble image (BGR)
        force_dark: If True, treat as dark bubble regardless of detection
        custom_color: Custom color (B, G, R) - None = auto-detect
        
    Returns:
        tuple: (processed_image, contour, is_dark, detected_color)
    """
    # Auto-detect background color if no custom_color provided
    if custom_color is None:
        detected_color = get_bubble_background_color(image)
    else:
        detected_color = custom_color
    
    if force_dark or is_dark_bubble(image):
        processed, contour, color_used = process_dark_bubble(image, detected_color)
        return processed, contour, True, color_used
    else:
        processed, contour, color_used = process_bubble(image, detected_color)
        return processed, contour, False, color_used


def process_bubble_preserve_gradient(image, text_mask=None):
    """
    Process speech bubble while preserving gradient/complex backgrounds.
    Only removes text, keeps original background using inpainting.
    
    Args:
        image: Input bubble image (BGR)
        text_mask: Mask of text region to remove (None = auto-detect)
        
    Returns:
        tuple: (processed_image, contour)
    """
    if text_mask is None:
        # Auto-detect text using edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text based on contrast
        bg_color = get_bubble_background_color(image)
        bg_intensity = np.mean(bg_color)
        
        if bg_intensity > 128:
            # Light background, dark text
            _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            # Dark background, light text
            _, text_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Use inpainting to remove text and preserve background
    # Dilate mask slightly to ensure complete text removal
    kernel = np.ones((3, 3), np.uint8)
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Inpaint to fill text region with surrounding background
    result = cv2.inpaint(image, text_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    # Find bubble contour
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
    
    return result, largest_contour

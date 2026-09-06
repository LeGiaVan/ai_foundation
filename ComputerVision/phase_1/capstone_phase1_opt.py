"""
QC Image Preprocessing Pipeline - Optimized Version
Features:
- Type hints for better readability
- Configurable parameters
- Optimized performance
- Better error handling
- Modular design
- Comprehensive logging
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ThresholdConfig:
    """Configuration for threshold-based defect detection"""
    clahe_clip_limit: float = 1.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    blur_kernel: Tuple[int, int] = (9, 9)
    morph_kernel: Tuple[int, int] = (9, 9)
    morph_iterations: int = 1
    min_area: int = 300
    max_area_ratio: float = 0.08  # % of image area
    adaptive_block_size: int = 51
    adaptive_c: int = 10
    lighting_grid: int = 4
    lighting_std_threshold: float = 8.0


@dataclass
class CannyConfig:
    """Configuration for edge-based defect detection"""
    blur_kernel: Tuple[int, int] = (9, 9)
    threshold1: int = 10
    threshold2: int = 30
    morph_kernel: Tuple[int, int] = (7, 7)
    morph_iterations: int = 2
    min_area: int = 50


@dataclass
class DebugConfig:
    """Debug output configuration"""
    save_debug: bool = False
    prefix: str = "debug"
    show_preview: bool = False


@dataclass
class Defect:
    """Defect information container"""
    area_px: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    perimeter_px: float
    contour: np.ndarray
    centroid: Optional[Tuple[float, float]] = None
    circularity: Optional[float] = None
    
    def __post_init__(self):
        """Calculate additional features if not provided"""
        if self.centroid is None:
            M = cv2.moments(self.contour)
            if M['m00'] != 0:
                self.centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        
        if self.circularity is None and self.perimeter_px > 0:
            self.circularity = (4 * np.pi * self.area_px) / (self.perimeter_px ** 2)
            # (4 * pi * pi * r^2) / (2 * pi * r)^2 = 1 - Hình tròn
            # Hình vuông: (4 * a^2) / (4a)^2 = 0.25

@dataclass
class PipelineResult:
    """Pipeline result container"""
    gray: np.ndarray
    binary: Optional[np.ndarray] = None
    defects: List[Defect] = field(default_factory=list)
    # Trong Python nếu khai báo defects: List[Defect] = []
    # Nó sẽ chỉ được tạo ra đúng 1 lần duy nhất trên RAM lúc Python biên dịch file.
    # Sau này nếu có nhiều Object PipelineResult thì nó cũng chỉ ghi vào 1 defect List duy nhất => Sai logic
    # Vì thế ta dùng field(default_factory=list) để tạo instance riêng cho mỗi object
    method_used: str = ""
    processing_time: float = 0.0
    is_defective: bool = False
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger('QC_Pipeline')
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_lighting_uneven(
    gray: np.ndarray, 
    grid: int = 4, 
    std_threshold: float = 8.0
) -> bool:
    """
    Detect if lighting is uneven across the image.
    
    Args:
        gray: Grayscale image
        grid: Number of grid divisions
        std_threshold: Standard deviation threshold
    
    Returns:
        True if lighting is uneven
    """
    h, w = gray.shape
    grid_h, grid_w = h // grid, w // grid
    
    # Vectorized computation for better performance
    means = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * grid_h, (i + 1) * grid_h
            x0, x1 = j * grid_w, (j + 1) * grid_w
            # Use faster mean calculation
            means.append(np.mean(gray[y0:y1, x0:x1]))
    
    return float(np.std(means)) > std_threshold


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 1.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE contrast enhancement"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def apply_blur(
    gray: np.ndarray,
    kernel: Tuple[int, int] = (5, 5),
    sigma: float = 0
) -> np.ndarray:
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(gray, kernel, sigma)


def apply_morphology(
    binary: np.ndarray,
    kernel_size: Tuple[int, int],
    operation: int,
    iterations: int = 1
) -> np.ndarray:
    """Apply morphological operation"""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(binary, operation, kernel, iterations=iterations)


def compute_defect_features(cnt: np.ndarray) -> Dict:
    """
    Compute defect features from contour
    
    Returns:
        Dictionary with area, bbox, perimeter, centroid
    """
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Compute centroid
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    else:
        cx, cy = x + w / 2, y + h / 2
    
    return {
        'area': area,
        'bbox': (x, y, w, h),
        'perimeter': perimeter,
        'centroid': (cx, cy)
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def qc_pipeline(
    image_path: Union[str, Path],
    mode: str = 'threshold',
    config: Optional[Dict] = None,
    debug_config: Optional[DebugConfig] = None
) -> PipelineResult:
    """
    Optimized QC Pipeline for defect detection.
    
    Args:
        image_path: Path to input image
        mode: 'threshold' for blob defects, 'canny' for edge defects
        config: Configuration dictionary (uses defaults if None)
        debug_config: Debug configuration
    
    Returns:
        PipelineResult containing defects and metadata
    """
    import time
    start_time = time.time()
    
    # Load configuration
    if config is None:
        config = {}
    
    if mode == 'threshold':
        cfg = ThresholdConfig(**config.get('threshold', {}))
    elif mode == 'canny':
        cfg = CannyConfig(**config.get('canny', {}))
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'threshold' or 'canny'")
    
    # Setup debug
    if debug_config is None:
        debug_config = DebugConfig()
    
    # ========================================================================
    # STEP 1: Load Image
    # ========================================================================
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    logger.info(f"Loaded image: {image_path}, shape: {gray.shape}")
    
    # ========================================================================
    # STEP 2: Process based on mode
    # ========================================================================
    binary = None
    defects = []
    contours = []
    method_used = ""
    
    if mode == 'threshold':
        binary, defects, method_used = _process_threshold(
            gray, cfg, debug_config
        )
    else:  # canny
        binary, defects, method_used = _process_canny(
            gray, cfg, debug_config
        )
    
    # ========================================================================
    # STEP 3: Prepare result
    # ========================================================================
    result = PipelineResult(
        gray=gray,
        binary=binary,
        defects=defects,
        method_used=method_used,
        processing_time=time.time() - start_time,
        is_defective=len(defects) > 0,
        metadata={
            'image_shape': gray.shape,
            'mode': mode,
            'defect_count': len(defects),
            'config': config
        }
    )
    
    logger.info(f"Pipeline completed: {len(defects)} defects found in {result.processing_time:.3f}s")
    
    # Save debug images if requested
    if debug_config.save_debug:
        _save_debug_images(gray, binary, result, debug_config.prefix)
    
    if debug_config.show_preview:
        _show_preview(result)
    
    return result


def _process_threshold(
    gray: np.ndarray,
    cfg: ThresholdConfig,
    debug_config: DebugConfig
) -> Tuple[np.ndarray, List[Defect], str]:
    """Process image using threshold-based method"""
    logger.debug("Processing with threshold method")
    
    # Step 1: Enhance contrast
    enhanced = apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_size)
    logger.debug("CLAHE applied")
    
    # Step 2: Blur to reduce noise
    blurred = apply_blur(enhanced, cfg.blur_kernel)
    logger.debug(f"Blur applied: {cfg.blur_kernel}")
    
    # Step 3: Choose threshold method
    uneven = is_lighting_uneven(
        gray, 
        grid=cfg.lighting_grid,
        std_threshold=cfg.lighting_std_threshold
    )
    
    if uneven:
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=cfg.adaptive_block_size,
            C=cfg.adaptive_c
        )
        method_used = f"adaptive (blockSize={cfg.adaptive_block_size})"
        logger.info(f"Lighting uneven detected → Using adaptive threshold")
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        method_used = "otsu"
        logger.info("Lighting uniform → Using Otsu threshold")
    
    # Step 4: Morphological cleaning
    binary_cleaned = apply_morphology(
        binary,
        cfg.morph_kernel,
        cv2.MORPH_OPEN,
        cfg.morph_iterations
    )
    logger.debug("Morphological opening applied")
    
    # Step 5: Extract defects (check both polarities)
    img_area = gray.shape[0] * gray.shape[1]
    max_area = int(img_area * cfg.max_area_ratio)
    
    defects = _extract_defects_from_binary(
        binary_cleaned,
        cfg.min_area,
        max_area
    )
    
    # Also check inverted binary
    defects_inv = _extract_defects_from_binary(
        cv2.bitwise_not(binary_cleaned),
        cfg.min_area,
        max_area
    )
    defects.extend(defects_inv)
    
    logger.info(f"Found {len(defects)} defects (min_area={cfg.min_area}, max_area={max_area})")
    
    return binary_cleaned, defects, method_used


def _process_canny(
    gray: np.ndarray,
    cfg: CannyConfig,
    debug_config: DebugConfig
) -> Tuple[np.ndarray, List[Defect], str]:
    """Process image using Canny edge detection"""
    logger.debug("Processing with Canny method")
    
    # Step 1: Blur (stronger to reduce noise)
    blurred = apply_blur(gray, cfg.blur_kernel)
    logger.debug(f"Strong blur applied: {cfg.blur_kernel}")
    
    # Step 2: Canny edge detection
    binary = cv2.Canny(blurred, cfg.threshold1, cfg.threshold2)
    logger.debug(f"Canny applied: t1={cfg.threshold1}, t2={cfg.threshold2}")
    
    # Step 3: Morphological closing to connect edges
    binary_cleaned = apply_morphology(
        binary,
        cfg.morph_kernel,
        cv2.MORPH_CLOSE,
        cfg.morph_iterations
    )
    logger.debug("Morphological closing applied")
    
    # Step 4: Extract defects
    defects = _extract_defects_from_binary(
        binary_cleaned,
        cfg.min_area,
        float('inf')  # No max area limit for canny
    )
    
    logger.info(f"Found {len(defects)} edge defects (min_area={cfg.min_area})")
    
    return binary_cleaned, defects, f"canny ({cfg.threshold1}/{cfg.threshold2})"


def _extract_defects_from_binary(
    binary: np.ndarray,
    min_area: float,
    max_area: float
) -> List[Defect]:
    """Extract defects from binary image with area filtering"""
    defects = []
    
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area < max_area:
            features = compute_defect_features(cnt)
            defect = Defect(
                area_px=features['area'],
                bbox=features['bbox'],
                perimeter_px=features['perimeter'],
                contour=cnt,
                centroid=features['centroid']
            )
            defects.append(defect)
    
    return defects


def _save_debug_images(
    gray: np.ndarray,
    binary: np.ndarray,
    result: PipelineResult,
    prefix: str
) -> None:
    """Save debug images"""
    cv2.imwrite(f"{prefix}_gray.jpg", gray)
    if binary is not None:
        cv2.imwrite(f"{prefix}_binary.jpg", binary)
    
    # Save annotated image if defects exist
    if result.defects:
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for defect in result.defects:
            x, y, w, h = defect.bbox
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(
                img_color,
                f"{defect.area_px:.0f}px",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        cv2.imwrite(f"{prefix}_defects.jpg", img_color)
    
    logger.debug(f"Debug images saved with prefix: {prefix}")


def _show_preview(result: PipelineResult) -> None:
    """Show preview of results"""
    img = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
    for defect in result.defects:
        x, y, w, h = defect.bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow("Defect Detection Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def draw_defects(
    image_path: Union[str, Path],
    result: PipelineResult,
    output_path: Union[str, Path] = "result_annotated.jpg",
    show_labels: bool = True
) -> str:
    """
    Draw defects on original image with bounding boxes and labels.
    
    Args:
        image_path: Original image path
        result: PipelineResult from qc_pipeline
        output_path: Output image path
        show_labels: Whether to show area/perimeter labels
    
    Returns:
        Path to saved annotated image
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    # Draw contours in green
    contours = [d.contour for d in result.defects]
    if contours:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    
    # Draw bounding boxes in red with labels
    for defect in result.defects:
        x, y, w, h = defect.bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        if show_labels:
            label = f"{defect.area_px:.0f}px"
            if defect.circularity:
                label += f" | C:{defect.circularity:.2f}"
            
            cv2.putText(
                img,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
    
    cv2.imwrite(str(output_path), img)
    logger.info(f"Annotated image saved to: {output_path}")
    
    return str(output_path)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    result = qc_pipeline(
        image_path=r"D:\ai_foundation\ComputerVision\phase_1\qc_test_dataset\qc_test_dataset\images\07_stain.jpg",
        mode='threshold'
    )
    print(f"Found {len(result.defects)} defects")
    
    # Example 2: With custom configuration
    # custom_config = {
    #     'threshold': {
    #         'min_area': 500,
    #         'max_area_ratio': 0.05,
    #         'clahe_clip_limit': 1.5
    #     }
    # }
    # result = qc_pipeline(
    #     image_path="D:\ai_foundation\ComputerVision\phase_1\images\image.png",
    #     mode='threshold',
    #     config=custom_config,
    #     debug_config=DebugConfig(save_debug=True, prefix='custom')
    # )
    
    # Example 3: Draw defects
    draw_defects(r"D:\ai_foundation\ComputerVision\phase_1\qc_test_dataset\qc_test_dataset\images\07_stain.jpg", result, "output.jpg")
    
    # Example 4: Canny mode
    # result = qc_pipeline(
    #     image_path="scratch.jpg",
    #     mode='canny'
    # )
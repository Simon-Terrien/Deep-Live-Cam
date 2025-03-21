from typing import Any, List, Tuple
import cv2
import numpy as np
import os
import logging
import threading
from modules.typing import Frame, Face
from modules.core import update_status
from modules.face_analyser import get_one_face
import modules.globals

# Setup logging
logger = logging.getLogger("DLC.HAIR-TRANSFER")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

NAME = "DLC.HAIR-TRANSFER"
THREAD_LOCK = threading.Lock()

# Constants for hair segmentation
HAIR_CLASS = 17  # Class ID for hair in segmentation models
FACIAL_HAIR_CLASS = 18  # Class ID for facial hair/beard

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

# Global variables
FACE_PARSER = None
FACE_PARSER_RESOLUTION = (512, 512)  # Resolution for face parsing model input


def pre_check() -> bool:
    """
    Check if required models exist and download if necessary
    
    Returns:
        bool: True if models are ready
    """
    logger.info("Running pre-check for Hair Transfer module")
    download_directory_path = models_dir
    
    if not os.path.exists(models_dir):
        logger.info(f"Creating models directory: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
    
    # In a complete implementation, you would download face parsing models here
    
    return True


def pre_start() -> bool:
    """
    Verify that valid source and target are selected
    
    Returns:
        bool: True if source and target are valid
    """
    logger.info("Running pre-start validation for Hair Transfer")
    
    if not modules.globals.source_path:
        logger.error("No source path selected")
        update_status("Select an image for source path.", NAME)
        return False
        
    if not modules.globals.target_path:
        logger.error("No target path selected")
        update_status("Select an image or video for target path.", NAME)
        return False
    
    # Check if source has a face
    source_img = cv2.imread(modules.globals.source_path)
    if source_img is None:
        logger.error(f"Could not read source image: {modules.globals.source_path}")
        update_status("Could not read source image.", NAME)
        return False
        
    source_face = get_one_face(source_img)
    if source_face is None:
        logger.error("No face detected in source image")
        update_status("No face detected in source image.", NAME)
        return False
    
    logger.info("Hair Transfer pre-start validation successful")
    return True


class SimulatedFaceParser:
    """
    A placeholder class to simulate face parsing functionality
    In a real implementation, you would use an actual neural network model
    """
    
    def __call__(self, image):
        """
        Simulate face parsing by using color-based and edge detection to create masks
        This is a simplified version - real implementation would use a neural network
        
        Args:
            image: Input face image
            
        Returns:
            A segmentation map with different regions
        """
        # Create an empty segmentation map
        seg_map = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to detect potential hair regions (dark areas at top of image)
        _, hair_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Create a rough hair region (top part of face)
        height, width = image.shape[:2]
        hair_region = np.zeros_like(hair_thresh)
        hair_region[0:int(height * 0.4), :] = 255
        
        # Combine threshold with region to get hair mask
        hair_mask = cv2.bitwise_and(hair_thresh, hair_region)
        
        # Create a beard region (bottom part of face, center area)
        beard_region = np.zeros_like(hair_thresh)
        beard_region[int(height * 0.7):, int(width * 0.3):int(width * 0.7)] = 255
        
        # Apply different threshold for beard detection
        _, beard_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        beard_mask = cv2.bitwise_and(beard_thresh, beard_region)
        
        # Set the hair class in the segmentation map
        seg_map[hair_mask > 0] = HAIR_CLASS
        
        # Set the facial hair class in the segmentation map
        seg_map[beard_mask > 0] = FACIAL_HAIR_CLASS
        
        return seg_map


def get_face_parser() -> Any:
    """
    Initialize and return a face parsing model
    
    Returns:
        The initialized face parser model
    """
    global FACE_PARSER
    
    with THREAD_LOCK:
        if FACE_PARSER is None:
            logger.info("Initializing face parsing model")
            try:
                # For this implementation, we'll use a simulated parser
                # In a real implementation, you would load a proper segmentation model
                # Such as BiSeNet, FaceParseNet, or similar
                
                FACE_PARSER = SimulatedFaceParser()
                
                logger.info("Face parsing model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing face parsing model: {str(e)}")
                logger.debug("Error details:", exc_info=True)
                raise
                
    return FACE_PARSER


def extract_hair_mask(face_img: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hair and facial hair masks from a face image
    
    Args:
        face_img: Face image
        
    Returns:
        Tuple containing (hair mask, facial hair mask)
    """
    logger.debug("Extracting hair masks")
    
    # Resize for consistent processing
    resized_img = cv2.resize(face_img, FACE_PARSER_RESOLUTION)
    
    # Get face parser
    face_parser = get_face_parser()
    
    # Get segmentation map
    seg_map = face_parser(resized_img)
    
    # Extract hair and facial hair masks
    hair_mask = (seg_map == HAIR_CLASS).astype(np.uint8) * 255
    facial_hair_mask = (seg_map == FACIAL_HAIR_CLASS).astype(np.uint8) * 255
    
    # Apply morphological operations to refine masks
    kernel = np.ones((5, 5), np.uint8)
    
    hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)
    hair_mask = cv2.GaussianBlur(hair_mask, (9, 9), 0)
    
    facial_hair_mask = cv2.dilate(facial_hair_mask, kernel, iterations=1)
    facial_hair_mask = cv2.GaussianBlur(facial_hair_mask, (7, 7), 0)
    
    # Resize masks back to original size
    hair_mask = cv2.resize(hair_mask, (face_img.shape[1], face_img.shape[0]))
    facial_hair_mask = cv2.resize(facial_hair_mask, (face_img.shape[1], face_img.shape[0]))
    
    return hair_mask, facial_hair_mask


def match_hair_color(source_img: Frame, target_img: Frame, mask: np.ndarray) -> Frame:
    """
    Match the hair color from source to target image
    
    Args:
        source_img: Source face image
        target_img: Target face image
        mask: Hair mask to apply color matching
        
    Returns:
        Color-matched image
    """
    # Check if mask has any hair pixels
    if np.sum(mask) == 0:
        return target_img
    
    # Convert to LAB color space for better color matching
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    
    # Normalize mask to range 0-1 for blending
    normalized_mask = mask.astype(float) / 255.0
    # Expand mask to 3 channels
    normalized_mask = np.repeat(normalized_mask[:, :, np.newaxis], 3, axis=2)
    
    # Calculate mean and std of source hair color (only where mask > 0)
    source_mask = cv2.resize(mask, (source_lab.shape[1], source_lab.shape[0])) if source_lab.shape[:2] != mask.shape[:2] else mask
    source_mask_3ch = np.repeat(source_mask[:, :, np.newaxis], 3, axis=2) if len(source_mask.shape) == 2 else source_mask
    
    # Avoid division by zero by checking if there are any masked pixels
    if np.sum(source_mask_3ch) > 0:
        # Calculate statistics only for masked pixels
        source_pixels = source_lab[source_mask_3ch > 0].reshape(-1, 3)
        source_mean = np.mean(source_pixels, axis=0)
        source_std = np.std(source_pixels, axis=0)
        
        target_pixels = target_lab[normalized_mask > 0.1].reshape(-1, 3)
        if len(target_pixels) > 0:
            target_mean = np.mean(target_pixels, axis=0)
            target_std = np.std(target_pixels, axis=0) + 1e-6  # Avoid division by zero
            
            # Create color transferred target image
            result_lab = np.copy(target_lab)
            
            # Only process pixels where mask is > 0
            mask_indices = normalized_mask > 0.1
            for i in range(3):
                channel = result_lab[:, :, i]
                channel_flat = channel[mask_indices[:, :, i]]
                # Apply color transfer
                adjusted = ((channel_flat - target_mean[i]) * (source_std[i] / target_std[i])) + source_mean[i]
                # Clip values to valid range
                adjusted = np.clip(adjusted, 0, 255)
                channel[mask_indices[:, :, i]] = adjusted
                result_lab[:, :, i] = channel
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
            
            # Create a feathered blend between original and color matched
            feathered_mask = cv2.GaussianBlur(normalized_mask, (21, 21), 0)
            blended = target_img * (1 - feathered_mask) + result_bgr * feathered_mask
            
            return blended.astype(np.uint8)
    
    return target_img


def transfer_hair(source_face_img: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """
    Transfer hair from source face to target face
    
    Args:
        source_face_img: Image of the source face
        target_frame: Full target frame
        target_face: Face object for the target
        
    Returns:
        Frame with hair transferred
    """
    # Get the bounding box from target face
    if target_face is None:
        return target_frame
        
    x_min, y_min, x_max, y_max = map(int, target_face.bbox)
    
    # Extract target face crop
    target_face_crop = target_frame[y_min:y_max, x_min:x_max]
    if target_face_crop.size == 0:
        logger.warning("Empty target face crop, skipping hair transfer")
        return target_frame
    
    # Resize source face to match target face size
    try:
        resized_source = cv2.resize(source_face_img, (target_face_crop.shape[1], target_face_crop.shape[0]))
    except Exception as e:
        logger.error(f"Error resizing source face: {str(e)}")
        return target_frame
    
    # Extract hair masks for both source and target
    source_hair_mask, source_beard_mask = extract_hair_mask(resized_source)
    target_hair_mask, target_beard_mask = extract_hair_mask(target_face_crop)
    
    # Apply color matching to hair and beard regions
    logger.debug("Applying hair color transfer")
    target_face_with_hair = match_hair_color(resized_source, target_face_crop, source_hair_mask)
    target_face_with_hair_beard = match_hair_color(resized_source, target_face_with_hair, source_beard_mask)
    
    # Create result frame by copying original frame
    result_frame = target_frame.copy()
    
    # Paste the modified face back into the frame
    result_frame[y_min:y_max, x_min:x_max] = target_face_with_hair_beard
    
    return result_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    """
    Process a single frame to transfer hair from source to target
    
    Args:
        source_face: Source face
        temp_frame: Frame to process
        
    Returns:
        Processed frame with hair transferred
    """
    if source_face is None:
        logger.warning("No source face provided, skipping hair transfer")
        return temp_frame
        
    try:
        # Get source face image
        source_img = cv2.imread(modules.globals.source_path)
        if source_img is None:
            logger.error(f"Could not read source image: {modules.globals.source_path}")
            return temp_frame
            
        x_min, y_min, x_max, y_max = map(int, source_face.bbox)
        if x_min < 0 or y_min < 0 or x_max > source_img.shape[1] or y_max > source_img.shape[0]:
            logger.error("Invalid source face bounding box")
            return temp_frame
            
        source_face_img = source_img[y_min:y_max, x_min:x_max]
        if source_face_img.size == 0:
            logger.error("Empty source face crop")
            return temp_frame
        
        # Get target face
        target_face = get_one_face(temp_frame)
        if target_face:
            logger.debug("Face detected in target, applying hair transfer")
            result_frame = transfer_hair(source_face_img, temp_frame, target_face)
            return result_frame
        else:
            logger.debug("No face detected in target frame, skipping hair transfer")
            return temp_frame
            
    except Exception as e:
        logger.error(f"Error in hair transfer: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        return temp_frame


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Process a single image for hair transfer
    
    Args:
        source_path: Path to source image
        target_path: Path to target image
        output_path: Path to save output image
    """
    logger.info(f"Processing image: {target_path}")
    
    try:
        # Read source image and get source face
        source_img = cv2.imread(source_path)
        source_face = get_one_face(source_img)
        
        if source_face is None:
            logger.error("No face detected in source image")
            return
            
        # Read target image
        target_img = cv2.imread(target_path)
        if target_img is None:
            logger.error(f"Failed to read target image: {target_path}")
            return
            
        # Process frame
        result = process_frame(source_face, target_img)
        
        # Save result
        cv2.imwrite(output_path, result)
        logger.info(f"Hair transfer result saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.debug("Error details:", exc_info=True)


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    """
    Process multiple frames for hair transfer
    
    Args:
        source_path: Path to source image
        temp_frame_paths: List of paths to frames
        progress: Progress tracker
    """
    logger.info(f"Processing {len(temp_frame_paths)} frames for hair transfer")
    
    try:
        # Read source image and get source face
        source_img = cv2.imread(source_path)
        if source_img is None:
            logger.error(f"Failed to read source image: {source_path}")
            return
            
        source_face = get_one_face(source_img)
        
        if source_face is None:
            logger.error("No face detected in source image")
            return
            
        # Process each frame
        for temp_frame_path in temp_frame_paths:
            try:
                # Read frame
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is None:
                    logger.warning(f"Failed to read frame: {temp_frame_path}")
                    continue
                    
                # Process frame
                result = process_frame(source_face, temp_frame)
                
                # Save result
                cv2.imwrite(temp_frame_path, result)
                
                if progress:
                    progress.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing frame {temp_frame_path}: {str(e)}")
                
        logger.info("Hair transfer completed for all frames")
        
    except Exception as e:
        logger.error(f"Error in frame processing: {str(e)}")
        logger.debug("Error details:", exc_info=True)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Process video frames for hair transfer
    
    Args:
        source_path: Path to source image
        temp_frame_paths: List of paths to video frames
    """
    logger.info(f"Processing video with {len(temp_frame_paths)} frames")
    
    # Use the core module to process the video with multi-threading
    from modules.processors.frame.core import process_video
    process_video(source_path, temp_frame_paths, process_frames)
    
    logger.info("Video processing completed")
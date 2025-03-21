from typing import Any, List
import cv2
import threading
import gfpgan
import os
import logging

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

# Setup logging
logger = logging.getLogger("DLC.FACE-ENHANCER")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    """
    Check if model exists and download if necessary
    
    Returns:
        bool: True if model is ready
    """
    logger.info("Running pre-check for Face Enhancer")
    download_directory_path = models_dir
    
    if not os.path.exists(models_dir):
        logger.info(f"Creating models directory: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
    if os.path.exists(model_path):
        logger.info(f"Model file already exists at: {model_path}")
    else:
        logger.info("Model file not found, downloading...")
        conditional_download(
            download_directory_path,
            [
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            ],
        )
        logger.info("Model download completed")
    
    return True


def pre_start() -> bool:
    """
    Verify that a valid target is selected
    
    Returns:
        bool: True if target is valid
    """
    logger.info("Running pre-start validation")
    
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        logger.error("No valid target path selected")
        update_status("Select an image or video for target path.", NAME)
        return False
        
    logger.info(f"Target validation successful: {modules.globals.target_path}")
    return True


def get_face_enhancer() -> Any:
    """
    Initialize and return the GFPGAN face enhancer model
    
    Returns:
        GFPGANer: The initialized face enhancer model
    """
    global FACE_ENHANCER
    
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            logger.info("Initializing Face Enhancer model")
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file missing: {model_path}")
                
            try:
                match platform.system():
                    case "Darwin":  # Mac OS
                        if torch.backends.mps.is_available():
                            logger.info("Using MPS (Metal Performance Shaders) for acceleration on macOS")
                            mps_device = torch.device("mps")
                            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=mps_device)  # type: ignore[attr-defined]
                        else:
                            logger.info("MPS not available, using CPU for macOS")
                            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
                    case _:  # Other OS
                        logger.info(f"Initializing on {platform.system()} with execution providers: {modules.globals.execution_providers}")
                        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
                        
                logger.info("Face Enhancer model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Face Enhancer: {str(e)}")
                logger.debug("Error details:", exc_info=True)
                raise

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    """
    Enhance a face in the given frame
    
    Args:
        temp_frame: The frame containing a face to enhance
        
    Returns:
        Frame: The enhanced frame
    """
    with THREAD_SEMAPHORE:
        logger.debug("Enhancing face")
        try:
            _, _, temp_frame = get_face_enhancer().enhance(temp_frame, paste_back=True)
            logger.debug("Face enhancement completed")
        except Exception as e:
            logger.error(f"Error during face enhancement: {str(e)}")
            # Return original frame if enhancement fails
            pass
    return temp_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    """
    Process a single frame to enhance faces
    
    Args:
        source_face: Not used in face enhancement
        temp_frame: The frame to process
        
    Returns:
        Frame: The processed frame with enhanced faces
    """
    try:
        target_face = get_one_face(temp_frame)
        if target_face:
            logger.debug("Face detected, applying enhancement")
            temp_frame = enhance_face(temp_frame)
        else:
            logger.debug("No face detected, skipping enhancement")
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
    
    return temp_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Process multiple frames for face enhancement
    
    Args:
        source_path: Not used in face enhancement
        temp_frame_paths: Paths to frames to be processed
        progress: Progress bar for tracking
    """
    logger.info(f"Processing {len(temp_frame_paths)} frames")
    
    for temp_frame_path in temp_frame_paths:
        try:
            logger.debug(f"Processing frame: {temp_frame_path}")
            temp_frame = cv2.imread(temp_frame_path)
            
            if temp_frame is None:
                logger.warning(f"Failed to read frame: {temp_frame_path}")
                continue
                
            result = process_frame(None, temp_frame)
            cv2.imwrite(temp_frame_path, result)
            
            if progress:
                progress.update(1)
                
        except Exception as e:
            logger.error(f"Error processing frame {temp_frame_path}: {str(e)}")
            logger.debug("Error details:", exc_info=True)
            
    logger.info("Finished processing frames")


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Process a single image for face enhancement
    
    Args:
        source_path: Not used in face enhancement
        target_path: Path to the target image
        output_path: Path where the enhanced image will be saved
    """
    logger.info(f"Processing image: {target_path}")
    
    try:
        target_frame = cv2.imread(target_path)
        
        if target_frame is None:
            logger.error(f"Failed to read target image: {target_path}")
            return
            
        result = process_frame(None, target_frame)
        cv2.imwrite(output_path, result)
        logger.info(f"Enhanced image saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.debug("Error details:", exc_info=True)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Process video frames for face enhancement
    
    Args:
        source_path: Not used in face enhancement
        temp_frame_paths: Paths to video frames to be processed
    """
    logger.info(f"Processing video with {len(temp_frame_paths)} frames")
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
    logger.info("Video processing completed")


def process_frame_v2(temp_frame: Frame) -> Frame:
    """
    Alternative process frame method for live mode
    
    Args:
        temp_frame: The frame to process
        
    Returns:
        Frame: The processed frame with enhanced faces
    """
    try:
        target_face = get_one_face(temp_frame)
        if target_face:
            logger.debug("Face detected in live mode, applying enhancement")
            temp_frame = enhance_face(temp_frame)
        else:
            logger.debug("No face detected in live mode, skipping enhancement")
    except Exception as e:
        logger.error(f"Error in live mode frame processing: {str(e)}")
        
    return temp_frame
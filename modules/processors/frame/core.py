#!/usr/bin/env python3

import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals
from  modules.logging import logging

# Create logger for this module
logger = logging.getLogger("Processor.Core")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    """
    Dynamically loads a frame processor module
    
    Args:
        frame_processor: Name of the frame processor to load
        
    Returns:
        The loaded module
        
    Raises:
        SystemExit: If the module was not found or doesn't implement the required interface
    """
    logger.info(f"Attempting to load frame processor: {frame_processor}")
    try:
        module_path = f'modules.processors.frame.{frame_processor}'
        logger.debug(f"Module path: {module_path}")
        
        frame_processor_module = importlib.import_module(module_path)
        
        # Interface verification
        missing_methods = []
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.error(f"Module {frame_processor} doesn't implement the required interface. Missing methods: {', '.join(missing_methods)}")
            sys.exit(1)
            
        logger.info(f"Frame processor {frame_processor} loaded successfully")
        return frame_processor_module
        
    except ImportError as e:
        logger.error(f"Failed to load frame processor {frame_processor}: {str(e)}")
        logger.debug(f"Error details: {e}", exc_info=True)
        sys.exit(1)


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    """
    Gets the list of frame processor modules
    
    Args:
        frame_processors: List of processor names to load
        
    Returns:
        List of loaded modules
    """
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        logger.info(f"Initial loading of frame processors: {', '.join(frame_processors)}")
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    """
    Updates the list of frame processor modules based on UI selections
    
    Args:
        frame_processors: List of active processor names
    """
    global FRAME_PROCESSORS_MODULES
    logger.debug(f"Updating processors from UI: {frame_processors}")
    logger.debug(f"Current UI state: {modules.globals.fp_ui}")
    
    # Activate processors that are enabled in the UI but not in the list
    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in frame_processors:
            logger.info(f"Activating processor {frame_processor} from UI")
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)
                modules.globals.frame_processors.append(frame_processor)
                logger.debug(f"Processor {frame_processor} added to active list")
            except Exception as e:
                logger.error(f"Failed to activate processor {frame_processor}: {str(e)}")
        
        # Deactivate processors that are disabled in the UI
        if state == False:
            try:
                logger.info(f"Deactivating processor {frame_processor} from UI")
                frame_processor_module = load_frame_processor_module(frame_processor)
                if frame_processor_module in FRAME_PROCESSORS_MODULES:
                    FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                if frame_processor in modules.globals.frame_processors:
                    modules.globals.frame_processors.remove(frame_processor)
                logger.debug(f"Processor {frame_processor} removed from active list")
            except Exception as e:
                logger.debug(f"Error during deactivation of {frame_processor}: {str(e)}")


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """
    Process multiple frames in parallel
    
    Args:
        source_path: Path to the source image
        temp_frame_paths: List of paths to temporary frames
        process_frames: Frame processing function
        progress: Progress bar (optional)
    """
    logger.info(f"Parallel processing of {len(temp_frame_paths)} frames with {modules.globals.execution_threads} threads")
    
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        
        # Wait and handle errors
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing frame {i+1}: {str(e)}")
                logger.debug("Error details:", exc_info=True)
    
    logger.info("Parallel frame processing completed")


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    """
    Process a video frame by frame
    
    Args:
        source_path: Path to the source image
        frame_paths: List of paths to frames
        process_frames: Frame processing function
    """
    logger.info(f"Starting video processing: {len(frame_paths)} frames to process")
    
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        # Display processing information
        postfix_info = {
            'execution_providers': modules.globals.execution_providers, 
            'execution_threads': modules.globals.execution_threads, 
            'max_memory': modules.globals.max_memory
        }
        progress.set_postfix(postfix_info)
        logger.info(f"Processing configuration: {postfix_info}")
        
        try:
            multi_process_frame(source_path, frame_paths, process_frames, progress)
            logger.info("Video processing completed successfully")
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            logger.debug("Error details:", exc_info=True)
            raise
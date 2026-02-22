import logging
import psutil
import openvino as ov

logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    """
    Returns the optimal device available on the current machine using OpenVINO Core.
    Prioritizes Intel GPU if available, else falls back to CPU.
    """
    core = ov.Core()
    available_devices = core.available_devices
    logger.debug(f"OpenVINO Available Devices: {available_devices}")
    
    # Simple priority logic
    if 'CPU' in available_devices:
        logger.info("Selected OpenVINO Device: CPU (GPU disabled due to severe dynamic shape latency)")
        return 'CPU'
        
    # Fallback standard
    return 'CPU'

def get_physical_cpu_cores() -> int:
    """
    Returns the number of physical CPU cores.
    Avoids over-subscription caused by Hyper-Threading logical cores.
    """
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None or physical_cores < 1:
        # Fallback if psutil fails (e.g. some obscure OS container)
        logical_cores = psutil.cpu_count(logical=True)
        if logical_cores and logical_cores > 1:
            physical_cores = max(1, logical_cores // 2)
        else:
            physical_cores = 1
            
    logger.debug(f"Detected physical CPU cores: {physical_cores}")
    return physical_cores

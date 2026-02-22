import logging
import psutil
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Strict order of priority for Execution Providers
EP_PRIORITY_LIST = [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',
    'DmlExecutionProvider',
    'CPUExecutionProvider'
]

def get_optimal_providers() -> list[str]:
    """
    Returns the optimal list of execution providers available on the current machine
    based on the strict priority list.
    """
    available_providers = ort.get_available_providers()
    logger.debug(f"ORT Available Providers: {available_providers}")
    
    # Filter the priority list so we only request what is actually available 
    # and strictly in the order we defined.
    optimal_providers = [ep for ep in EP_PRIORITY_LIST if ep in available_providers]
    
    if 'CPUExecutionProvider' not in optimal_providers:
        optimal_providers.append('CPUExecutionProvider')
        
    logger.info(f"Selected ORT Execution Providers: {optimal_providers}")
    return optimal_providers

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

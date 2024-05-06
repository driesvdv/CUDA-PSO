import numpy as np

class ObjectiveFunction:
    
    def __init__(self) -> None:
        pass

    def calculate_price(offset_1, device_1, pricing):
    
        device_usage = device_1
        price_frame = pricing[offset_1:offset_1 + len(device_usage)]
    
        return np.sum(device_usage * price_frame)
    
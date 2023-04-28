from abc import abstractmethod

class Batch:
    """
    Класс батча даных для тех случаев когда необходимо переопределить стандартное поведение
    """
    def __init__(self, data):
        self._data = data
    
    def data(self):
        """Return samples data"""
        return self._data
   
    # Abstract method(s)
    @abstractmethod
    def __len__(self) -> int:
        """Return samples count in current batch"""
        
    @abstractmethod
    def to(self, device) -> None:
        """Send currect batch to device"""

    
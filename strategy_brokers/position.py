from dataclasses import dataclass

@dataclass
class Position:
    """
    Represents a single, live position in a specific instrument.
    """
    instrument: str
    quantity: int
    average_price: float = 0.0

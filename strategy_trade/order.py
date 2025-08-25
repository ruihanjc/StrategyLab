from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class OrderStatus(Enum):
    PENDING = "Pending"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"

@dataclass
class Order:
    """
    Represents a single trade order for live execution.
    """
    instrument: str
    quantity: int
    order_type: OrderType
    status: OrderStatus = field(default=OrderStatus.PENDING)
    limit_price: Optional[float] = None
    order_id: Optional[int] = None

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders must have a limit_price.")

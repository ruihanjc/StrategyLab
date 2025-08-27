from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class OrderStatus(Enum):
    PENDING_SUBMIT = "PendingSubmit"
    PENDING_CANCEL = "PendingCancel"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    API_CANCELLED = "ApiCancelled"
    CANCELLED = "Cancelled"
    FILLED = "Filled"
    INACTIVE = "Inactive"

IB_ORDER_STATUS_MAP = {
    "PendingSubmit": OrderStatus.PENDING_SUBMIT,
    "PendingCancel": OrderStatus.PENDING_CANCEL,
    "PreSubmitted": OrderStatus.PRE_SUBMITTED,
    "Submitted": OrderStatus.SUBMITTED,
    "ApiCancelled": OrderStatus.API_CANCELLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Filled": OrderStatus.FILLED,
    "Inactive": OrderStatus.INACTIVE,
}

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
    status: OrderStatus = field(default=OrderStatus.PRE_SUBMITTED)
    limit_price: Optional[float] = None
    order_id: Optional[int] = None

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders must have a limit_price.")

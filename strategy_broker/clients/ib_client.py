import logging

from ib_insync import ib
from strategy_broker.ib_connection import IBConnection
from ib_insync import Contract




IB_ERROR_TYPES = {
    200: "ambgious_contract",
    501: "already connected",
    502: "can't connect",
    503: "TWS need upgrading",
    100: "Max messages exceeded",
    102: "Duplicate ticker",
    103: "Duplicate orderid",
    104: "can't modify filled order",
    105: "trying to modify different order",
    106: "can't transmit orderid",
    107: "can't transmit incomplete order",
    109: "price out of range",
    110: "tick size wrong for price",
    122: "No request tag has been found for order",
    123: "invalid conid",
    133: "submit order failed",
    134: "modify order failed",
    135: "cant find order",
    136: "order cant be cancelled",
    140: "size should be an integer",
    141: "price should be a double",
    201: "order rejected",
    202: "order cancelled",
}

IB_IS_ERROR = list(IB_ERROR_TYPES.keys())


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class IBClient(object):

    def __init__(self):

        setup_logging()
        self.ib = IBConnection().connect()
        self.logger = logging.getLogger(__name__)

    @property
    def ib_connection(self) -> IBConnection:
        return self.ib

    @property
    def client_id(self) -> int:
        return self.ib.client.clientId

    def error_handler(
        self, reqid: int, error_code: int, error_string: str, ib_contract: Contract
    ):
        """
        Error handler called from server
        Needs to be attached to ib connection

        :param reqid: IB reqid
        :param error_code: IB error code
        :param error_string: IB error string
        :param contract: IB contract or None
        :return: success
        """

        msg = "Reqid %d: %d %s for %s" % (
            reqid,
            error_code,
            error_string,
            str(ib_contract),
        )

        iserror = error_code in IB_IS_ERROR
        if iserror:
            # Serious requires some action
            myerror_type = IB_ERROR_TYPES.get(error_code, "generic")
            self.broker_error(msg=msg,  log=self.logger, myerror_type=myerror_type)

        else:
            # just a general message
            self.broker_message(msg=msg, log=self.logger)

    def broker_error(self, msg, log, myerror_type):
        log.warning(msg)

    def broker_message(self, log, msg):
        log.debug(msg)

    def refresh(self):
        self.ib.sleep(0.00001)


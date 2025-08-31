from strategy_core.sysobjects import Instrument
from ..extractors import *
from ..extractors.client_extractor.ibkr_equity_extractor import IBKREquityExtractor
from ..extractors.rest_extractor.alphavantage_extractor import AlphaVantageExtractor
from ..extractors.rest_extractor.marketstack_extractor import MarketStackExtractor


class RequesterFactory:

    @staticmethod
    def create(instrument : Instrument, api_config) -> BaseExtractor:
        match instrument.asset_class:
            case "equity":
                match instrument.source:
                    case "MarketStack":
                        return MarketStackExtractor(instrument, api_config)
                    case "AlphaVantage":
                        return AlphaVantageExtractor(instrument, api_config)
                    case "IBKR":
                        return IBKREquityExtractor(instrument)
                    case _:
                        raise RuntimeError("Failed to choose equity extractor")
            case _:
                raise RuntimeError("Failed to choose extractor")

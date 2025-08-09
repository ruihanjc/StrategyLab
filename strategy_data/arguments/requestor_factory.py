from strategy_core.sysobjects import Instrument
from ..extractors import *
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
                    case _:
                        raise RuntimeError("Failed to choose equity extractor")
            case _:
                raise RuntimeError("Failed to choose extractor")

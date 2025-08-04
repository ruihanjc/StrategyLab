from ..extractors import *

from market_data.extractors import BaseExtractor
from market_data.extractors.rest_extractor.alphavantage_extractor import AlphaVantageExtractor
from market_data.extractors.rest_extractor.marketstack_extractor import MarketStackExtractor


class RequesterFactory:

    @staticmethod
    def create(entry, api_config) -> BaseExtractor:
        match entry[0]:
            case "equity":
                match entry[1]:
                    case "MarketStack":
                        return MarketStackExtractor(entry, api_config)
                    case "AlphaVantage":
                        return AlphaVantageExtractor(entry, api_config)
                    case _:
                        raise RuntimeError("Failed to choose equity extractor")
            case _:
                raise RuntimeError("Failed to choose extractor")

from Extractor import *

class RequesterFactory:

    def create(settings, api_config: object) -> BaseExtractor:
        match settings.source:
            case "MarketStack":
                return MarketStackExtractor(settings, api_config)
            case "AlphaVantage":
                return AlphaVantageExtractor(settings, api_config)
            case _:
                return "Something's wrong with the configs"


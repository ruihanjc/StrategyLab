class Strategy:
    """
    Abstract base class for all trading strategies
    Now with YAML configuration support
    """

    def __init__(self, trading_rules: list):
        """
        Initialize strategy with optional config

        Parameters:
        -----------
        config: dict or str
            Either a config dict or a string with strategy name to load from YAML
        """
        self.parameters = {}
        self.trading_rules = trading_rules

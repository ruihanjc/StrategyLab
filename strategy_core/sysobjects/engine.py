class TradingEngine:

    def __init__(self, portfolio, strategy, *args, **kwargs):
        self.portfolio = portfolio
        self.strategy = strategy
        self.data = args[0]

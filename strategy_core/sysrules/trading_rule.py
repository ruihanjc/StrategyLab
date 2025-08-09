class TradingRule:
    """
    Base class for trading rules
    """

    def __init__(self, rule_function, params, data=None):
        if data is None:
            data = []
        self.rule_function = rule_function
        self.params = params
        self.data = data

    def get_data(self):
        return self.data

    def get_rule(self):
        return self.rule_function

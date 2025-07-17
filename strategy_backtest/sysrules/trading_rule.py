
from strategy_backtest.sysutils.constants import arg_not_supplied

from strategy_backtest.sysrules.trading_rule_components import (
    get_trading_rule_components
)

# Trading rules container


class TradingRule(object):

    def __init__(self, trading_rule, data : list = arg_not_supplied, other : dict = arg_not_supplied):
        trading_rule_components = get_trading_rule_components(trading_rule, data, other)

        self.rule = trading_rule_components.rule
        self.data = trading_rule_components.data
        self.other_data = trading_rule_components.other_data
        self.data_args = trading_rule_components.data_args




# Global functions


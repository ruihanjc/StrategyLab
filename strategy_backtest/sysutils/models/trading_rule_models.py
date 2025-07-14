
from strategy_backtest.sysutils.object import (
    resolve_rule,
    hasallattr
)

from strategy_backtest.sysutils.constants import *


class TradingRuleComponents(object):
    def __init__(self, rule, data, other_data, data_args: list = arg_not_supplied):
        rule, data, other_data, data_args = self.pre_process(rule, data, other_data)

        self._rule = rule
        self._data = data
        self._other_data = other_data
        self._data_args = data_args



    @staticmethod
    def pre_process(rule, data, other_data, data_args: list = arg_not_supplied, ):
        rule = resolve_rule(rule)

        if isinstance(data, str):
            # turn into a 1 item list or wont' get parsed properly
            data = [data]

        if len(data) == 0:
            if data is not arg_not_supplied:
                print("WARNING! Ignoring data_args as data is list length zero")
            # if no data provided defaults to using price
            data = [DEFAULT_PRICE_SOURCE]
            data_args = [{}]

        return rule, data, other_data, data_args

    @property
    def rule(self):
        return self.rule

    @property
    def data(self) -> list:
        return self.data

    @property
    def data_args(self) -> list:
        return self.data_args

    @property
    def other_data(self) -> dict:
        return self.other_data



def get_trading_rule_components(rule, data, other):
    if data is arg_not_supplied:
        data = []

    if other is arg_not_supplied:
        other = {}

    if _already_a_trading_rule(rule):
        # looks like it is already a trading rule
        rule_components = _create_rule_from_existing_rule(rule, data=data, other_args=other)
    else:
        rule_components = _create_rule_from_tuple(rule, data=data, other_args=other)

    return rule_components


def _create_rule_from_existing_rule(
    rule, data: list, other_args: dict
) -> TradingRuleComponents:
    _throw_warning_if_passed_rule_and_data(
        "tradingRule", data=data, other_args=other_args
    )
    return TradingRuleComponents(
        rule = rule.rule,
        data = rule.data,
        other_data = rule.other_data,
        data_args= rule.data_args
    )


def _throw_warning_if_passed_rule_and_data(
    type_of_rule_passed: str, data: list, other_args: dict
):
    if len(data) > 0 or len(other_args) > 0:
        print(
            "WARNING: Creating trade rule with 'rule' type %s argument, ignoring data and/or other args"
            % type_of_rule_passed
        )


def _create_rule_from_tuple(rule, data, other_args):
    return TradingRuleComponents(rule, data, other_args)

def _already_a_trading_rule(rule):
    return hasallattr(rule, ["function", "data", "other_args"])
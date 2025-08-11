"""
Per-Rule Parameter Optimization Usage Examples
每个交易规则的参数优化使用示例
"""

def example_ewmac_optimization():
    """
    EWMAC规则优化示例
    """
    print("EWMAC规则优化示例:")
    print("""
    # 1. 导入EWMAC优化函数
    from strategy_core.sysrules.ewmac import optimize_ewmac, apply_ewmac_params
    
    # 2. 运行优化 (假设你已经有了配置好的engine)
    trials, best_params = optimize_ewmac(
        engine=your_engine,
        target_metric='sharpe_ratio',
        max_evals=50
    )
    
    # 3. 应用最佳参数
    apply_ewmac_params(your_engine, best_params)
    
    # 4. 运行最终回测
    final_results = your_engine.run()
    """)


def example_ma_crossover_optimization():
    """
    移动平均交叉规则优化示例
    """
    print("移动平均交叉规则优化示例:")
    print("""
    # 1. 导入MA Crossover优化函数
    from strategy_core.sysrules.ma_crossover import optimize_ma_crossover, apply_ma_crossover_params
    
    # 2. 运行优化
    trials, best_params = optimize_ma_crossover(
        engine=your_engine,
        target_metric='sharpe_ratio', 
        max_evals=30
    )
    
    # 3. 应用最佳参数
    apply_ma_crossover_params(your_engine, best_params)
    
    # 4. 运行最终回测
    final_results = your_engine.run()
    """)


def example_mixed_strategy_optimization():
    """
    混合策略优化示例 - 多个规则分别优化
    """
    print("混合策略优化示例:")
    print("""
    # 如果你的策略包含多个不同类型的规则，可以分别优化
    
    from strategy_core.sysrules.ewmac import optimize_ewmac, apply_ewmac_params
    from strategy_core.sysrules.ma_crossover import optimize_ma_crossover, apply_ma_crossover_params
    
    # 1. 优化EWMAC规则
    ewmac_trials, ewmac_best = optimize_ewmac(your_engine, 'sharpe_ratio', 30)
    apply_ewmac_params(your_engine, ewmac_best)
    
    # 2. 优化MA Crossover规则
    ma_trials, ma_best = optimize_ma_crossover(your_engine, 'sharpe_ratio', 30)
    apply_ma_crossover_params(your_engine, ma_best)
    
    # 3. 运行最终回测
    final_results = your_engine.run()
    
    print(f"EWMAC最佳参数: {ewmac_best}")
    print(f"MA Crossover最佳参数: {ma_best}")
    """)


def create_new_rule_optimization_template():
    """
    新规则优化模板
    """
    print("创建新规则优化的模板:")
    print("""
    # 在你的新规则文件中添加以下函数:
    
    def optimize_your_rule(engine, target_metric='sharpe_ratio', max_evals=50):
        '''你的规则参数优化'''
        try:
            from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        except ImportError:
            print("需要安装hyperopt: pip install hyperopt")
            return None, None
        
        import numpy as np
        
        # 定义你的参数空间
        space = {
            "engine": engine,
            "param1": hp.quniform("param1", min_val, max_val, step),
            "param2": hp.quniform("param2", min_val, max_val, step),
        }
        
        # 保存原始参数
        original_params = {}
        for i, rule in enumerate(engine.strategy.trading_rules):
            if rule.get_rule().__name__ == 'your_rule_name':
                original_params[i] = rule.params.copy()
        
        def optimization_objective(params):
            try:
                param1 = int(params["param1"])
                param2 = int(params["param2"])
                
                # 检查参数有效性 (根据你的规则调整)
                if param1 >= param2:  # 例如
                    return {"loss": float('inf'), "status": STATUS_OK}
                
                # 只更新你的规则的参数
                for rule in engine.strategy.trading_rules:
                    if rule.get_rule().__name__ == 'your_rule_name':
                        rule.params.update({"param1": param1, "param2": param2})
                
                # 运行回测
                results = engine.run()
                metric_value = results.metrics.get(target_metric, 0.0)
                
                if not np.isfinite(metric_value):
                    metric_value = -999.0
                
                print(f"YOUR_RULE: param1={param1}, param2={param2} -> {target_metric}={metric_value:.4f}")
                
                return {"loss": -metric_value, "status": STATUS_OK}
                
            except Exception as e:
                print(f"YOUR_RULE优化错误: {str(e)}")
                return {"loss": float('inf'), "status": STATUS_OK}
        
        print(f"🚀 开始YOUR_RULE参数优化，目标: {target_metric}")
        print("-" * 50)
        
        trials = Trials()
        best = fmin(
            fn=optimization_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        # 恢复原始参数
        for i, rule in enumerate(engine.strategy.trading_rules):
            if i in original_params:
                rule.params.update(original_params[i])
        
        best_params = {
            'param1': int(best['param1']),
            'param2': int(best['param2'])
        }
        
        best_metric = -min(trials.losses())
        
        print("-" * 50)
        print(f"🎉 YOUR_RULE优化完成! 最佳{target_metric}: {best_metric:.4f}")
        print(f"📊 最佳参数: {best_params}")
        
        return trials, best_params
    
    
    def apply_your_rule_params(engine, best_params):
        '''应用你的规则最佳参数'''
        print(f"🔧 应用YOUR_RULE最佳参数: {best_params}")
        
        for rule in engine.strategy.trading_rules:
            if rule.get_rule().__name__ == 'your_rule_name':
                rule.params.update(best_params)
        
        print("✅ YOUR_RULE参数已应用到策略!")
    """)


if __name__ == "__main__":
    print("📊 Per-Rule Parameter Optimization Examples")
    print("=" * 60)
    
    example_ewmac_optimization()
    print("\n" + "=" * 60)
    
    example_ma_crossover_optimization()
    print("\n" + "=" * 60)
    
    example_mixed_strategy_optimization()
    print("\n" + "=" * 60)
    
    create_new_rule_optimization_template()
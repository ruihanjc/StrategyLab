import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class ChartGenerator:
    @staticmethod
    def create_candlestick(df: pd.DataFrame, title: str) -> go.Figure:
        """Create candlestick chart with volume"""
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Volume bar chart
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=title,
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    @staticmethod
    def create_returns_analysis(df: pd.DataFrame) -> dict:
        """Create returns analysis charts"""
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()

        # Returns distribution
        returns_dist = go.Figure(
            data=[go.Histogram(x=df['daily_return'], nbinsx=50)],
            layout=dict(title='Returns Distribution',
                        xaxis_title='Daily Returns',
                        yaxis_title='Frequency')
        )

        # Returns QQ plot
        returns_qq = px.scatter(
            x=df['daily_return'].sort_values(),
            y=pd.qcut(df['daily_return'].rank(method='first'),
                      q=100, labels=False) / 100,
            labels={'x': 'Returns', 'y': 'Probability'},
            title='Returns Q-Q Plot'
        )
        returns_qq.add_trace(
            go.Scatter(x=[-0.1, 0.1], y=[0, 1],
                       mode='lines', name='Normal')
        )

        return {
            'distribution': returns_dist,
            'qq_plot': returns_qq
        }


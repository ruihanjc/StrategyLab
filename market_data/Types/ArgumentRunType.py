import argparse

class ArgumentRunType:

    def __init__(self):
        self.parser = self.setup_parser()

    @staticmethod
    def setup_parser():
        """Setup argument parser"""
        parser = argparse.ArgumentParser(description='Market Data Pipeline')

        # Add run arguments
        parser.add_argument('-service', type=str,
                            help='Service type (e.g., Equity)')

        parser.add_argument('-source', type=str,
                            help='Data source (e.g., MarketStack)')

        parser.add_argument('-ticker', type=str,
                            help='Stock ticker symbol')

        return parser

    def parse_arguments(self, args=None):
        """Parse command line arguments"""
        if args:
            # If args is a string, split it into a list
            if isinstance(args, str):
                # Split by space but respect quotes
                import shlex
                args = shlex.split(args)
            return self.parser.parse_args(args)
        return self.parser.parse_args()



    @staticmethod
    def validate_arguments(args):
        """Validate required arguments"""
        required = ['service', 'source', 'ticker']
        missing = [arg for arg in required if getattr(args, arg) is None]

        if missing:
            print(f"Error: Missing required arguments: {', '.join(missing)}")
            return False
        return True
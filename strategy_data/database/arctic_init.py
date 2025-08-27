
import pandas as pd
from datetime import datetime
import os
import logging

from strategy_data.database.arctic_connection import get_arcticdb_connection


class ArcticDBInitializer:
    def __init__(self):
        self.logger = None
        self.setup_logging()
        project_dir = os.path.abspath(__file__ + "/../../../")
        self.arctic_path = os.path.join(project_dir, 'arcticdb')
        self.arctic = get_arcticdb_connection(self.arctic_path)
        self.setup_logging()

    def setup_logging(self):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        logs_dir = os.path.join(project_dir, 'logs')
        try:
            os.makedirs(logs_dir, exist_ok=True)
            log_filename = os.path.join(logs_dir, f'arctic_init_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_filename)
                ]
            )
        except PermissionError:
            # Fall back to console-only logging if file logging fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
        self.logger = logging.getLogger(__name__)

    def initialize_connection(self):
        """Initialize ArcticDB connection"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.arctic_path, exist_ok=True)
            self.logger.info(f"Initialized directory: {self.arctic_path}")

            # Connect to ArcticDB
            arctic = get_arcticdb_connection(self.arctic_path)
            self.logger.info("Connected to ArcticDB successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ArcticDB: {str(e)}")
            return False

    def create_libraries(self):
        """Create required libraries if they don't exist"""
        required_libraries = {
            'equity': 'Stock market equities data',
            'market_index': 'Market index data',
            'crypto': 'Cryptocurrency market data',
            'forex': 'Foreign exchange market data',
            'futures': 'Futures market data',
            'metadata': 'Market data metadata'
        }

        created = []
        existing = []

        for lib_name, description in required_libraries.items():
            try:
                if not self.arctic.has_library(lib_name):
                    self.arctic.create_library(lib_name)
                    created.append(lib_name)
                    self.logger.info(f"Created library: {lib_name}")
                else:
                    existing.append(lib_name)
                    self.logger.info(f"Library already exists: {lib_name}")
            except Exception as e:
                self.logger.error(f"Error creating library {lib_name}: {str(e)}")

        return created, existing

    def verify_libraries(self):
        """Verify all libraries are accessible"""
        libraries = self.arctic.list_libraries()
        print(libraries)
        print(self.arctic.get_uri())
        self.logger.info("\nVerifying libraries:")

        for lib_name in libraries:
            try:
                lib = self.arctic.get_library(lib_name)
                symbols = lib.list_symbols()
                self.logger.info(f"Library {lib_name}: {len(symbols)} symbols")
            except Exception as e:
                self.logger.error(f"Error accessing library {lib_name}: {str(e)}")

    def create_test_data(self):
        """Create test data if libraries are empty"""
        try:
            lib = self.arctic.get_library('equity')
            if not lib.list_symbols():
                # Create sample data
                dates = pd.date_range(start='2024-01-01', periods=10)
                test_data = pd.DataFrame({
                    'date': dates,
                    'open': range(100, 110),
                    'high': range(105, 115),
                    'low': range(95, 105),
                    'close': range(102, 112),
                    'volume': range(1000, 1010),
                    'ticker': ['TEST'] * 10,
                    'service': ['Equity'] * 10,
                    'source': ['Test'] * 10,
                    'timestamp': [datetime.now()] * 10
                })

                lib.write('TEST_SYMBOL', test_data)
                self.logger.info("Created test data in equity library")
        except Exception as e:
            self.logger.error(f"Error creating test data: {str(e)}")

    def initialize(self):
        """Run full initialization process"""
        if not self.initialize_connection():
            return False

        created, existing = self.create_libraries()

        self.logger.info("\nInitialization Summary:")
        if created:
            self.logger.info(f"Created libraries: {', '.join(created)}")
        if existing:
            self.logger.info(f"Existing libraries: {', '.join(existing)}")

        self.verify_libraries()

        return True


def arctic_init():
    # Use relative path from script location
    initializer = ArcticDBInitializer()
    success = initializer.initialize()

    if success:
        print("\nArcticDB initialization completed successfully!")
        print("Check arctic_init.log for details")
    else:
        print("\nArcticDB initialization failed!")
        print("Check arctic_init.log for error details")


if __name__ == "__main__":
    arctic_init()
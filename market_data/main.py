from Types.ArgumentRunType import ArgumentRunType
from config_manager import ConfigManager
import logging
import sys
from requestor_factory import RequesterFactory
from ArcticDB import arcticdb_writer


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main(config_arguments):
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        logger.info("Initializing configuration...")
        config = ConfigManager()
        api_config = config.get_api_config()
        database_config = config.get_database_config()

        logger.info(f"API Configuration loaded with timeout: {api_config['timeout']}")
        logger.info("Database configuration loaded")

        # Parse and validate arguments
        logger.info("Parsing command line arguments...")
        argument_parser = ArgumentRunType()

        # Skip the first argument (script name) if it's sys.argv
        if isinstance(config_arguments, list) and config_arguments[0].endswith('.py'):
            parsed_args = argument_parser.parse_arguments(config_arguments[1:])
        else:
            parsed_args = argument_parser.parse_arguments(config_arguments)

        if not argument_parser.validate_arguments(parsed_args):
            logger.error("Argument validation failed")
            return False

        # Create requester and fetch data
        logger.info(f"Creating requester for service: {parsed_args.service}, "
                    f"source: {parsed_args.source}, ticker: {parsed_args.ticker}")
        requestor = RequesterFactory.create(parsed_args, api_config)

        logger.info("Fetching data...")
        fetched_data = requestor.run()

        if fetched_data is None:
            logger.error("No data fetched from the source")
            return False

        # Store data in ArcticDB
        logger.info("Initializing ArcticDB storage...")
        arcticdb_helper = arcticdb_writer.MarketDataStore(database_config)

        logger.info("Storing fetched data...")
        success = arcticdb_helper.store_market_data(fetched_data)

        if success:
            logger.info("Data successfully stored in ArcticDB")
            return True
        else:
            logger.error("Failed to store data in ArcticDB")
            return False

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)
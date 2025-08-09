from arguments.argument_runtype import ArgumentRunType
import logging
import sys
from database.arcticdb_writer import ArcticWriter
from strategy_data.arguments.requestor_factory import RequesterFactory
from strategy_data.configuration.config_manager import ConfigManager


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def standard_update_data(config_arguments):
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        logger.info("Initializing configuration...")
        config = ConfigManager()
        api_config = config.get_api_config()
        database_config = config.get_database_config()
        update_config = config.get_daily_update_config()

        logger.info(f"API Configuration loaded with timeout: {api_config['timeout']}")
        logger.info("database configuration loaded")

        # Parse and validate arguments
        logger.info("Parsing command line arguments...")
        argument_parser = ArgumentRunType()

        # Skip the first argument (script name) if it's sys.argv
        if isinstance(config_arguments, list) and config_arguments[0].endswith('.py'):
            parsed_args = argument_parser.parse_arguments(config_arguments[1:])
        else:
            parsed_args = argument_parser.parse_arguments(config_arguments)

        update_list = argument_parser.validate_arguments(parsed_args, update_config)

        if not update_list:
            logger.error("Argument validation failed")
            return False

        # Create requester and fetch data
        logger.info(f"Creating requester for service: {update_list}")

        for instrument in update_list:
            requester = RequesterFactory.create(instrument, api_config)

            logger.info("Fetching data...")
            fetched_data = requester.run()

            if fetched_data is None:
                logger.info(f"Already up to date for instrument {instrument}")
                continue

            # Store data in database
            logger.info("Initializing database storage...")
            arctic_writer = ArcticWriter()
            if_ingested = arctic_writer.store_market_data(fetched_data)

            logger.info(f"Storing fetched data for {instrument}")

            if if_ingested:
                logger.info("Data successfully stored in database")
            else:
                logger.error("Failed to store data in database")

        return True
    except Exception as error:
        logger.error(f"Application error: {str(error)}", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        success = standard_update_data(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)

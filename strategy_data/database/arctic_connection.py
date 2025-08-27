import logging
import arcticdb

logger = logging.getLogger(__name__)


class ArcticConnection:
    """
    Singleton pattern for ArcticDB connection to prevent multiple connections
    in the same process.
    """
    _instance = None
    _libraries = {}
    _libpath = None

    @classmethod
    def get_instance(cls, lib_path):
        """Get the ArcticDB connection instance or create it if it doesn't exist"""
        if cls._instance is None:

            logger.info(f"Initializing ArcticDB connection to: {lib_path}")
            try:
                arctic_dir = f"lmdb://{lib_path}"
                cls._instance = arcticdb.Arctic(arctic_dir)
                logger.info("ArcticDB connection established successfully")
            except Exception as e:
                logger.error(f"Failed to create ArcticDB connection: {str(e)}")
                raise
        return cls._instance

    @classmethod
    def close(cls):
        """Close the ArcticDB connection if it exists"""
        if cls._instance is not None:
            logger.info("Closing ArcticDB connection")
            try:
                # ArcticDB might not have an explicit close method
                # But we'll reset our references
                pass
            except Exception as e:
                logger.warning(f"Error closing ArcticDB connection: {str(e)}")
            # Reset the instance
            cls._instance = None
            cls._libraries = {}
            logger.info("ArcticDB connection reset")


def get_arcticdb_connection(curPath):
    """Helper function to get ArcticDB connection"""
    return ArcticConnection.get_instance(curPath)


def close_arcticdb_connection():
    """Helper function to close ArcticDB connection"""
    ArcticConnection.close()
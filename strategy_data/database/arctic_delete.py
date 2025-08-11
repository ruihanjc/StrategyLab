from pathlib import Path
import os
import shutil
import logging
from datetime import datetime
import time
from strategy_data.database.arctic_connection import get_arcticdb_connection


class ArcticDBCleaner:
    def __init__(self):
        project_dir = os.path.abspath(__file__ + "/../../../")
        self.arctic_path = os.path.join(project_dir, 'arcticdb')
        self.arctic = get_arcticdb_connection(self.arctic_path)
        self.setup_logging()

    def setup_logging(self):
        log_filename = f'./logs/arctic_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_filename)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def backup_before_delete(self):
        """Create backup before deletion"""
        try:
            if os.path.exists(self.arctic_path):
                backup_path = f"{self.arctic_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(self.arctic_path, backup_path)
                self.logger.info(f"Created backup at: {backup_path}")
                return backup_path
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            return None

    def list_contents(self):
        """List all libraries and symbols before deletion"""
        try:
            if self.arctic:
                self.logger.info("\nCurrent ArcticDB Contents:")
                libraries = self.arctic.list_libraries()

                for lib_name in libraries:
                    lib = self.arctic.get_library(lib_name)
                    symbols = lib.list_symbols()
                    self.logger.info(f"\nLibrary: {lib_name}")
                    self.logger.info(f"Symbols: {symbols}")
        except Exception as e:
            self.logger.error(f"Error listing contents: {str(e)}")

    def cleanup(self, force=False):
        """Clean up ArcticDB"""
        try:
            # List contents before deletion
            self.list_contents()

            if not force:
                confirmation = input("\nAre you sure you want to delete the database? (yes/no): ")
                if confirmation.lower() != 'yes':
                    self.logger.info("Cleanup cancelled by user")
                    return False

            # Create backup
            backup_path = self.backup_before_delete()
            if not backup_path and not force:
                self.logger.error("Backup failed, aborting cleanup")
                return False

            # Disconnect from ArcticDB
            self.arctic = None
            time.sleep(1)  # Give time for connection to close

            # Delete the directory
            if os.path.exists(self.arctic_path):
                shutil.rmtree(self.arctic_path)
                self.logger.info(f"Deleted ArcticDB directory: {self.arctic_path}")
            else:
                self.logger.info("ArcticDB directory does not exist")

            return True

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ArcticDB Cleanup Tool')
    parser.add_argument('--force', action='store_true',
                        help='Force cleanup without confirmation')
    args = parser.parse_args()

    # Use relative path from script location
    cleaner = ArcticDBCleaner()

    if cleaner.cleanup(force=args.force):
        print("\nArcticDB cleanup completed successfully!")
        print("Check the log file for details")
    else:
        print("\nArcticDB cleanup failed or was cancelled!")
        print("Check the log file for details")


if __name__ == "__main__":
    main()

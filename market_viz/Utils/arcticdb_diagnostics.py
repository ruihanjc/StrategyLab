# arctic_diagnostic.py
import arcticdb as adb
import os
from pathlib import Path


class ArcticDiagnostic:
    def __init__(self, arctic_path):
        self.arctic_path = arctic_path
        self.arctic_uri = f"lmdb://{arctic_path}"

    def run_diagnostics(self):
        print("\n=== database Diagnostic Report ===")

        # Check path exists
        print("\n1. Checking database path:")
        path = Path(self.arctic_path)
        print(f"Path: {path.absolute()}")
        print(f"Exists: {path.exists()}")
        if path.exists():
            print(f"Contents: {os.listdir(path)}")

        # Try connecting
        print("\n2. Attempting database connection:")
        try:
            arctic = adb.Arctic(self.arctic_uri)
            print("Connection successful")

            # List libraries
            print("\n3. Checking libraries:")
            libraries = arctic.list_libraries()
            print(f"Libraries found: {libraries}")

            # If no libraries, create test library
            if not libraries:
                print("\n4. No libraries found. Creating test library...")
                try:
                    arctic.create_library('test_library')
                    print("Test library created successfully")
                    libraries = arctic.list_libraries()
                    print(f"Updated libraries: {libraries}")
                except Exception as e:
                    print(f"Error creating test library: {e}")

            # Check each library
            print("\n5. Checking each library:")
            for lib_name in libraries:
                try:
                    lib = arctic.get_library(lib_name)
                    symbols = lib.list_symbols()
                    print(f"\nLibrary: {lib_name}")
                    print(f"Symbols: {symbols}")
                except Exception as e:
                    print(f"Error accessing library {lib_name}: {e}")

        except Exception as e:
            print(f"Connection failed: {e}")

        # Check permissions
        print("\n6. Checking permissions:")
        try:
            test_file = path / "test_write.txt"
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("Write permissions OK")
        except Exception as e:
            print(f"Permission error: {e}")


if __name__ == "__main__":
    # Adjust this path to your database location
    arctic_path = "../../arcticdb"

    diagnostic = ArcticDiagnostic(arctic_path)
    diagnostic.run_diagnostics()
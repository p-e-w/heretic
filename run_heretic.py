import sys
import os

# Ensure the local version of the package is preferred
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from heretic.main import main
if __name__ == "__main__":
    main()

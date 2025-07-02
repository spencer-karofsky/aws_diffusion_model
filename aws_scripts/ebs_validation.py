"""
Execute unit tests defined in aws_setup/tests/ec2_test.py

Designed to be run by the CLI
"""
import unittest
import sys
import os

# Looks at project root directory; used for finding aws_setup/tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="aws_setup/tests", pattern="ebs_test.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
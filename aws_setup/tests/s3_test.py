"""
Test S3BucketManager and S3ObjectManager functionalities with moto (fake AWS calls that mimics boto3) and Python's unittest
"""
from core import *

class S3BucketManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked S3 and S3BucketManager"""
        self.mock_s3 = mock_aws(service="s3")
        self.mock_s3.start()

        self.client = boto3.client('s3')
        self.bucket_manager = S3BucketManager()

        self.fake_bucket_names = [f'my-fake-test-bucket-{i}' for i in range(10)]
    
    def tearDown(self):
        """Terminate test resources
        """
        self.mock_s3.stop()

    def _all_buckets_exist(self):
        """Return True if all fake buckets in fake_bucket_names exist
        """
        return all(
            [self.bucket_manager._check_bucket_exists(bucket_name)
             for bucket_name in self.fake_bucket_names]
        )
    
    def _no_buckets_exist(self):
        """Return True if no fake bucket in fake_bucket_names exists
        """
        return all(
            [not self.bucket_manager._check_bucket_exists(bucket_name)
             for bucket_name in self.fake_bucket_names]
        )
    
    def test_check_bucket_exists(self):
        """Test S3BucketManager._check_bucket_exists
        """
        # Use boto3 documentation for setup
        fake_bucket_name = 'my-fake-test-bucket-abc'
        self.client.create_bucket(Bucket=fake_bucket_name)

        self.assertTrue(self.bucket_manager._check_bucket_exists(fake_bucket_name))
        self.assertFalse(self.bucket_manager._check_bucket_exists(fake_bucket_name + '1'))

    def test_create_bucket(self):
        """Test S3BucketManager.create_bucket
        """
        # Ensure no buckets in fake_bucket_names exist
        self.assertTrue(self._no_buckets_exist())

        for bucket_name in self.fake_bucket_names:
            self.bucket_manager.create_bucket(bucket_name)

        # Ensure all buckets in fake_bucket_names exist
        self.assertTrue(self._all_buckets_exist())

        # Try creating buckets again, and ensure they fail
        for bucket_name in self.fake_bucket_names:
            self.assertFalse(self.bucket_manager.create_bucket(bucket_name))


    def test_delete_bucket(self):
        """Test S3BucketManager.delete_bucket
        """
        # Ensure all buckets in fake_bucket_names exist
        self.assertTrue(self._no_buckets_exist())

        # Create Buckets
        for bucket_name in self.fake_bucket_names:
            self.bucket_manager.create_bucket(bucket_name)

        # Delete all buckets in fake_news_bucket
        for bucket_name in self.fake_bucket_names:
            self.bucket_manager.delete_bucket(bucket_name)

        # Ensure no buckets in fake_bucket_names exist
        self.assertTrue(self._no_buckets_exist())

        # Try deleting buckets again, and ensure they fail
        for bucket_name in self.fake_bucket_names:
            self.assertFalse(self.bucket_manager.delete_bucket(bucket_name))


# Test S3ObjectManager
@mock_aws
class S3ObjectManagerTest(unittest.TestCase):
    pass
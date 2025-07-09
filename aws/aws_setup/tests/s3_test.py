"""
Test S3BucketManager and S3ObjectManager functionalities with moto (fake AWS calls that mimics boto3) and Python's unittest
"""
import boto3
from aws_setup.managers.s3_manager import S3BucketManager, S3ObjectManager
import unittest
from moto import mock_s3
import os
import tempfile

class S3BucketManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked S3BucketManager
        """
        self.mock = mock_s3()
        self.mock.start()

        self.client = boto3.client('s3')
        self.bucket_manager = S3BucketManager()

        self.fake_bucket_names = [f'my-fake-test-bucket-{i}' for i in range(10)]
    
    def tearDown(self):
        """Terminate test resources
        """
        self.mock.stop()

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

        # Create buckets
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
        # Ensure no buckets in fake_bucket_names exist
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
    
    def test_list_buckets(self):
        """Test S3BucketManager.list_buckets
        """
        # Assert no buckets exist
        self.assertTrue(self._no_buckets_exist())

        # Create buckets
        for bucket_name in self.fake_bucket_names:
            self.bucket_manager.create_bucket(bucket_name)

        # Retrieve bucket names and compare to fake_bucket_names
        buckets = self.bucket_manager.list_buckets()
        bucket_names = sorted([bucket['Name'] for bucket in buckets])

        self.assertListEqual(bucket_names, sorted(self.fake_bucket_names))


class S3ObjectManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked S3ObjectManager
        """
        self.mock = mock_s3()
        self.mock.start()

        self.client = boto3.client('s3')

        self.bucket_name = 'my-test-bucket'
        self.bucket = S3BucketManager().create_bucket(self.bucket_name)
        self.object_manager = S3ObjectManager()

        self.temp_objects = []
        self.object_names = []

        # Save objects/files and object/file names
        for i in range(10):
            temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            file_name = f'my-fake-test-object-{i}.txt'
            temp.write(f'hello, {file_name}')
            temp.close()

            self.temp_objects.append(temp.name)
            self.object_names.append(file_name)
    
    def tearDown(self):
        """Terminate test resources
        """
        for obj in self.temp_objects:
            os.remove(obj)

        self.mock.stop()
    
    def _all_objs_exist(self):
        """Return True if all objects exist in bucket: self.bucket_name
        """
        return all(
            [self.object_manager._check_object_exists(self.bucket_name, obj_name)
             for obj_name in self.object_names]
        )

    def _no_objs_exist(self):
        """Return True if no objects exist in bucket: self.bucket_name
        """
        return all(
            [not self.object_manager._check_object_exists(self.bucket_name, obj_name)
             for obj_name in self.object_names]
        )
    
    def test_upload_object(self):
        """Test S3ObjectManager.upload_object
        """
        # Check that no objects exist intially
        self.assertTrue(self._no_objs_exist())

        # Add objects to bucket
        for local_path, obj_key in zip(self.temp_objects, self.object_names):
            self.object_manager.upload_object(self.bucket_name, obj_key, local_path)
        
        # Check that all objects exist
        self.assertTrue(self._all_objs_exist())

        # Try to add objects again, and ensure they fail
        for local_path, obj_key in zip(self.temp_objects, self.object_names):
            self.assertFalse(self.object_manager.upload_object(self.bucket_name,
                                                               obj_key,
                                                               local_path))
    
    def test_download_object(self):
        """Test S3ObjectManager.download_object
        """
        # Ensure no objects in the bucket exist
        self.assertTrue(self._no_objs_exist())
        
        # Upload all objects to download
        for local_path, obj_key in zip(self.temp_objects, self.object_names):
            self.object_manager.upload_object(self.bucket_name, obj_key, local_path)
        
        # Download all objects
        with tempfile.TemporaryDirectory() as tmpdir:
            for obj_key, original_path in zip(self.object_names, self.temp_objects):
                destination_path = os.path.join(tmpdir, obj_key)
                self.object_manager.download_object(self.bucket_name, obj_key, destination_path)
                
                # Compare content of original and downloaded file
                with open(original_path, 'r') as original_file:
                    original_data = original_file.read()
                with open(destination_path, 'r') as downloaded_file:
                    downloaded_data = downloaded_file.read()
                
                self.assertEqual(original_data, downloaded_data, 
                                f'Contents of {obj_key} do not match after download')

    def test_delete_object(self):
        """Test S3ObjectManager.delete_object
        """
        # Assert all objects exist in the bucket
        # If no all objects exist, upload them
        if not self._all_objs_exist():
            for local_path, obj_key in zip(self.temp_objects, self.object_names):
                self.object_manager.upload_object(self.bucket_name,
                                                  obj_key,
                                                  local_path)

        # Delete all objects in the bucket
        for obj_key in self.object_names:
            self.object_manager.delete_object(self.bucket_name, obj_key)

        # Assert no objects exist in the bucket
        self.assertTrue(self._no_objs_exist())

    def test_list_objects(self):
        """Test S3ObjectManager.list_objects
        """
        # Assert no objects exist in the bucket
        self.assertTrue(self._no_objs_exist())

        # Upload objects in temp_objects
        for local_path, obj_key in zip(self.temp_objects, self.object_names):
            self.object_manager.upload_object(self.bucket_name,
                                              obj_key,
                                              local_path)
        
        # Get object names
        objs_in_bucket = self.object_manager.list_objects(self.bucket_name)
        obj_names = sorted([obj[0] for obj in objs_in_bucket])

        # Assert uploaded bucket's objects match object_names
        self.assertListEqual(obj_names, sorted(self.object_names))

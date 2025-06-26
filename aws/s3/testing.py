from create_bucket import BotoS3Client

names = ['ddpm-project-data',
         'ddpm-project-models',
         'ddpm-project-outputs']

s3 = BotoS3Client()

for name in names:
    s3.create_bucket(name)

print('Done')

from sagemaker.model import Model
import sagemaker

# Initialize SageMaker session
session = sagemaker.Session()
role = 'arn:aws:iam::463224263437:role/princecc'
bucket = 'spotify-recommendation-prince'
region = 'ap-south-1'

# Reference existing model
model = Model(
    image_uri=sagemaker.image_uris.retrieve('sklearn', region, version='0.23-1'),
    model_data='s3://spotify-recommendation-prince/output/sagemaker-scikit-learn-2025-05-15-11-00-27-528/output/model.tar.gz',
    role=role,
    sagemaker_session=session,
    source_dir='.',
    entry_point='inference.py'
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='music-recommendation-endpoint'
)
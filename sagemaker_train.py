import sagemaker
from sagemaker.estimator import Estimator
import boto3

# Initialize SageMaker session
session = sagemaker.Session()
role = 'arn:aws:iam::463224263437:role/princecc'  # Your role ARN
bucket = 'spotify-recommendation-prince'  # Replace with your bucket name
region = 'ap-south-1'

# Define training job
estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve('sklearn', region, version='0.23-1'),
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',  # Previously successful instance type
    output_path=f's3://{bucket}/output',
    entry_point='train.py',  # Specify the training script
    source_dir='.',
    hyperparameters={
        'input-dir': '/opt/ml/input/data/training',
        'output-dir': '/opt/ml/model'
    },
    sagemaker_session=session
)

# Specify input data
estimator.fit({'training': f's3://{bucket}/data/train_data.csv'})

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='spotify-recommendation-endpoint'
)
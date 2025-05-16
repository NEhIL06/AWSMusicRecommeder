import boto3
import json

def lambda_handler(event, context):
    client = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    endpoint_name = 'music-recommendation-endpoint'
    body = json.loads(event['body'])
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(body)
    )
    result = json.loads(response['Body'].read().decode())
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
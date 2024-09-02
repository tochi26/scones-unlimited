import json
import boto3
import base64
import logging

s3 = boto3.client('s3')

ENDPOINT = "image-classification-2024-08-31-05-42-07-605"
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Lambda function 1: Data Preprocessing (Assumed Example)
def image_serializer_lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-08-28-08-18-04-685"


# Lambda function 2: Model Inference
def image_classifier_lambda_handler(event, context):
    # Log the incoming event
    logger.info("Event: %s", json.dumps(event))

    image_data = event['Payload']['body']['image_data']
    key = event['Payload']['body']['s3_key']
    bucket = event['Payload']['body']['s3_bucket']

    try:
        # Decode the image data
        image = base64.b64decode(image_data)
    except Exception as e:
        return {
            'statusCode': 400,
            'body': {'error': f"Failed to decode base64 image: {str(e)}"}
        }

    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Make a prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image
    )

    # Get the inference results
    inferences = json.loads(response['Body'].read().decode('utf-8'))

    # We return the data back to the Step Function
    result = {
        'statusCode': 200,
        'body': {
            'image_data': image_data,
            'inferences': inferences,
            's3_key': key,
            's3_bucket': bucket
        }
    }
    return result


# Lambda function 3: Filter Low-Confidence Inferences
THRESHOLD = 0.93


def image_inference_lambda_handler(event, context):
    # Log the incoming event
    logger.info("Event: %s", json.dumps(event))

    # Extract the inferences from the event
    inferences = event['Payload']['body']['inferences']
    key = event['Payload']['body']['s3_key']
    bucket = event['Payload']['body']['s3_bucket']

    # Check if inferences is already a list
    if isinstance(inferences, list):
        inferences = [float(inf) for inf in inferences]
    else:
        # If it's not a list, assume it's a JSON string and parse it
        inferences = [float(inf) for inf in json.loads(inferences)]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(inference >= THRESHOLD for inference in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, return with an error status
    if meets_threshold:
        result = {
            'statusCode': 200,
            'body': {
                'image_data': event['Payload']['body']['image_data'],
                'inferences': inferences,
                'meets_threshold': True,
                's3_key': key,
                's3_bucket': bucket
            }
        }
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return result

# Add additional Lambda functions below as needed
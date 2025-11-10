import json
import boto3
import os
from strands import Agent
import uuid

model_id = os.environ.get("MODEL_ID", "")
s3_client = boto3.client('s3')

agent = Agent(model=model_id, callback_handler=None)

def validate_bank_statement(document: bytes) -> bool:

    validator_prompt = """
        Is this document a property tax statement? Please respond with "yes" or "no" only.
    """

    response = agent([
        {"text": validator_prompt},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": document,
                },
            },
        },
    ])
    output = str(response)
    print(f"Output: {output}")
    if output.strip().lower() == "yes":
        return True
    else:
        return False

def handler(event, context):
    """
    Lambda handler for Step Functions triggered by S3 PutObject events.
    
    Args:
        event: Step Functions event containing S3 object information
        context: Lambda context object
    
    Returns:
        dict: Contains the binary content of the uploaded document as bytes
    """
    try:
        # Extract S3 bucket and key from the event
        # The event structure will contain S3 event information
        if 'Records' in event:
            # Direct S3 event trigger
            s3_event = event['Records'][0]
            bucket = s3_event['s3']['bucket']['name']
            key = s3_event['s3']['object']['key']
        else:
            # Step Functions input format
            bucket = event.get('bucket')
            key = event.get('key')
        
        if not bucket or not key:
            raise ValueError("Missing bucket or key in event")
        
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read the binary content
        file_content = response['Body'].read()

        validation_result = validate_bank_statement(file_content)
        
        return {
            'bucket': bucket,
            'key': key,
            'valid': validation_result
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }


if __name__ == "__main__":
    # For local testing
    test_event = {
        'bucket': 'test-bucket',
        'key': 'test-document.png'
    }
    result = handler(test_event, None)
    print(json.dumps(result, default=str))

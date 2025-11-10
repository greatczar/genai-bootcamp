import json
import boto3
import os
from strands import Agent
import uuid
from pydantic import BaseModel, Field

class PropertyTaxStatement(BaseModel):
    PropertyTaxYear: int = Field(description="The year the property was assessed")
    PropertyTaxes: float = Field(description="The taxes of the property")


model_id = os.environ.get("MODEL_ID", "")
s3_client = boto3.client('s3')

agent = Agent(model=model_id, callback_handler=None)


def extract_bank_statement_data(document: bytes) -> dict:
    """
    Extracts structured data from a property tax statement image.
    """
    # extractor_prompt = """
    #     Please extract the following information from this bank statement and return it as a JSON object:
    #     - BankName
    #     - AccountNumber
    #     - OpeningBalance
    #     - ClosingBalance
    #     - StartDate
    #     - EndDate
    # """
    base_prompt = [
            {
                "text": "Extract all relevant information from the property tax statement provided.",
            },
            {
                "image": {
                    "format": "png",
                    "source": {
                        "bytes": document,
                    },
                },
            },
        ]
    response = agent.structured_output(
        PropertyTaxStatement,
        prompt=base_prompt
    )

    try:
        output_dict = response.model_dump()
        return output_dict

    except Exception as e:
        print(f"An error occurred during model conversion: {e}")
        return {}


def handler(event, context):
    """
    Lambda handler for document extraction.
    """
    print("Extracting document data...")
    print(f"Event: {json.dumps(event)}")
    
    bucket = event['bucket']
    key = event['key']
    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    
    # Read the binary content
    file_content = response['Body'].read()

    extracted_data = extract_bank_statement_data(file_content)

    is_valid = bool(extracted_data)

    return {
        'bucket': bucket,
        'key': key,
        'valid': is_valid,
        'extracted_data': extracted_data,
        'retry_count': event.get('retry_count', 0) + 1
    }

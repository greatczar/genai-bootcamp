from aws_cdk import BundlingOptions, Duration
from aws_cdk.aws_s3 import Bucket
from constructs import Construct
from aws_cdk.aws_dynamodb import TableV2
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_iam as iam

class Fn(Construct):
    function: _lambda.Function

    def __init__(self, scope: Construct, id: str,
                 code_path: str,
                 timeout: int = 30,
                 memory_size: int = 512,
                 bucket: Bucket | None = None,
                 ddb_table: TableV2 | None = None,
                ) -> None:
        super().__init__(scope, id)

        self.function = _lambda.Function(self, 'Fn',
                          timeout=Duration.seconds(timeout),
                          architecture=_lambda.Architecture.X86_64,
                          runtime=_lambda.Runtime.PYTHON_3_13,
                          code=_lambda.Code.from_asset(code_path,
                                    bundling=BundlingOptions(
                                        image=_lambda.Runtime.PYTHON_3_13.bundling_image,
                                        command=[
                                            'bash', '-c',
                                            'pip install uv && uv export --frozen --no-dev --no-editable -o requirements.txt && pip install -r requirements.txt -t /asset-output && cp -r *py /asset-output/'
                                        ],
                                        user='root'
                                    )
                          ),
                          handler="main.handler",
                          memory_size=memory_size,
                          environment={
                                    "MODEL_ID": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                          },
                        )
        # Add Bedrock permissions to the Lambda function
        self.function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:InvokeModel"
                ],
                resources=["*"]
            )
        )

        if (bucket):
            _ = bucket.grant_read_write(self.function)
        if (ddb_table):
            _ = self.function.add_environment('DDB_TABLE', ddb_table.table_name)
            _ = ddb_table.grant_read_write_data(self.function)


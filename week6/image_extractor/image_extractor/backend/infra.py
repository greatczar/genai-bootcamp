from aws_cdk import Duration, RemovalPolicy, BundlingOptions, Stack
from constructs import Construct
import aws_cdk.aws_dynamodb as dynamodb
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_notifications as s3n
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as tasks
import aws_cdk.aws_events as events
import aws_cdk.aws_events_targets as targets
import aws_cdk.aws_iam as iam
from .fn import Fn

class Backend(Construct):
    endpoint: str
    domain_name: str

    def __init__(self, scope: "Construct", id: str) -> None:
        super().__init__(scope, id)

        dynamodb_table = dynamodb.TableV2(self, 'ImageProcessorTable',
                                          partition_key=dynamodb.Attribute(name='PK', type=dynamodb.AttributeType.STRING),
                                          sort_key=dynamodb.Attribute(name='SK', type=dynamodb.AttributeType.STRING),
                                          removal_policy=RemovalPolicy.DESTROY,
                                         )
        input_bucket = s3.Bucket(self, 'InputBucket', event_bridge_enabled=True)

        validate_input_fn = Fn(self, 'ValidateInputFn',
                               code_path='image_extractor/backend/input_validator',
                               bucket=input_bucket,
                               timeout=120,
                               )
        
        extractor_fn = Fn(self, 'ExtractorFn',
                          code_path='image_extractor/backend/extractor',
                          bucket=input_bucket,
                          timeout=120)
        
        # Create output validator function stub
        output_validator_fn = Fn(self, 'OutputValidatorFn',
                                 code_path='image_extractor/backend/output_validator',
                                 timeout=30)
        
        # Step Functions tasks
        validate_input_task = tasks.LambdaInvoke(self, 'ValidateInputTask',
                                               lambda_function=validate_input_fn.function,
                                               output_path='$.Payload')
        
        extract_task = tasks.LambdaInvoke(self, 'ExtractTask',
                                        lambda_function=extractor_fn.function,
                                        output_path='$.Payload')
        
        validate_output_task = tasks.LambdaInvoke(self, 'ValidateOutputTask',
                                                lambda_function=output_validator_fn.function,
                                                output_path='$.Payload')
        
        # Success and failure states
        success_state = sfn.Succeed(self, 'SuccessState')
        failure_state = sfn.Fail(self, 'FailureState',
                               error='ValidationError',
                               cause='Document validation failed')
        
        # Choice state for validation
        validation_choice = sfn.Choice(self, 'IsValidDocument')
        validation_choice.when(sfn.Condition.boolean_equals('$.valid', True), extract_task)
        validation_choice.otherwise(failure_state)
        
        # Choice state for output validation
        output_validation_choice = sfn.Choice(self, 'IsValidOutput')
        output_validation_choice.when(sfn.Condition.boolean_equals('$.valid', True), success_state)
        output_validation_choice.when(sfn.Condition.number_equals('$.retry_count', 3), failure_state)
        output_validation_choice.otherwise(extract_task)
        
        # Connect the workflow
        validate_input_task.next(validation_choice)
        extract_task.next(validate_output_task)
        validate_output_task.next(output_validation_choice)
        
        # Create the state machine
        state_machine = sfn.StateMachine(self, 'DocumentProcessingWorkflow',
                                       definition=validate_input_task,
                                       timeout=Duration.minutes(5))
        
        
        s3_upload_rule = events.Rule(self, 'S3UploadRule',
                                   event_pattern=events.EventPattern(
                                       source=['aws.s3'],
                                       detail_type=['Object Created'],
                                       detail={
                                           'bucket': {
                                               'name': [input_bucket.bucket_name]
                                           },
                                           'object': {
                                               'key': [{'suffix': '.png'}, 
                                                    {'suffix': '.jpg'},
                                                    {'suffix': '.jpeg'}, 
                                                    {'suffix': '.webp'}]
                                           }
                                       }
                                   ))
        
        s3_upload_rule.add_target(targets.SfnStateMachine(state_machine,
                                                        input=events.RuleTargetInput.from_object({
                                                            'bucket': events.EventField.from_path('$.detail.bucket.name'),
                                                            'key': events.EventField.from_path('$.detail.object.key')
                                                        })))
        
        

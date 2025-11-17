from sys import prefix

from aws_cdk.aws_certificatemanager import Certificate
import aws_cdk as cdk
from aws_cdk import BundlingFileAccess, BundlingOptions, DockerImage, Duration, RemovalPolicy
from constructs import Construct
import json
import aws_cdk.aws_cloudfront as cloudfront
import aws_cdk.aws_cloudfront_origins as origins
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_deployment as s3deploy

class Frontend(Construct):
    domain_name: str

    def __init__(self, scope: Construct, id: str, 
                 backend_endpoint: str, 
                 admin_endpoint: str,
                 user_pool_id: str,
                 user_pool_client_id: str,
                 custom_certificate_arn: str|None = None,
                 custom_domain_name: str|None = None,
                 **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        frontend_bucket = s3.Bucket(self, 'FrontendBucket')
        admin_frontend_bucket = s3.Bucket(self, 'AdminFrontendBucket')
        admin_env_bucket = s3.Bucket(self, 'AdminEnvBucket')

        s3_origin = origins.S3BucketOrigin.with_origin_access_control(frontend_bucket,
                                                                       origin_access_levels=[cloudfront.AccessLevel.READ, cloudfront.AccessLevel.LIST],
                                                                      )
        admin_s3_origin = origins.S3BucketOrigin.with_origin_access_control(admin_frontend_bucket,
                                                                       origin_access_levels=[cloudfront.AccessLevel.READ, cloudfront.AccessLevel.LIST],
                                                                      )
        admin_env_s3_origin = origins.S3BucketOrigin.with_origin_access_control(admin_env_bucket,
                                                                       origin_access_levels=[cloudfront.AccessLevel.READ, cloudfront.AccessLevel.LIST],
                                                                      )
        origin_request_policy = cloudfront.OriginRequestPolicy(self, "OriginRequestPolicy",
                                                               cookie_behavior=cloudfront.OriginRequestCookieBehavior.all(),
                                                               query_string_behavior=cloudfront.OriginRequestQueryStringBehavior.all(),
                                                               )
        backend_origin = origins.HttpOrigin(
            domain_name=backend_endpoint,
            read_timeout=Duration.seconds(60),
            keepalive_timeout=Duration.seconds(60),
            connection_timeout=Duration.seconds(10),
        )

        admin_origin = origins.HttpOrigin(
            domain_name=admin_endpoint,
            read_timeout=Duration.seconds(11),
            keepalive_timeout=Duration.seconds(2),
            connection_timeout=Duration.seconds(5),
        )
        
        rewrite_function = cloudfront.Function(self, "RewriteFunction",
            code=cloudfront.FunctionCode.from_inline("""
function handler(event) {
    var request = event.request;
    var uri = request.uri;

    // If it is a path for the admin SPA and does not have a file extension in the last path segment,
    // rewrite to the admin index page to support client-side routing.
    if (uri.startsWith('/admin') && !/\\.[^/]+$/.test(uri)) {
        request.uri = '/admin/index.html';
    }
    return request;
}
"""))
        domain_names = []
        certificate = None
        if custom_certificate_arn and custom_domain_name:
            domain_names = [custom_domain_name]
            certificate = Certificate.from_certificate_arn(self, 'CustomCertificate', custom_certificate_arn)
        
        distribution = cloudfront.Distribution(self, 'Distribution',
                                                            certificate=certificate,
                                                            domain_names=domain_names,
                                                            default_root_object='index.html',
                                                            default_behavior=cloudfront.BehaviorOptions(
                                                                origin=s3_origin,
                                                                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                                                                function_associations=[cloudfront.FunctionAssociation(
                                                                    function=rewrite_function,
                                                                    event_type=cloudfront.FunctionEventType.VIEWER_REQUEST
                                                                )]
                                                            ),
                                                            additional_behaviors={
                                                                '/_app/env.js': cloudfront.BehaviorOptions(
                                                                    origin=admin_env_s3_origin,
                                                                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                                                                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED
                                                                ),
                                                                '/admin/*': cloudfront.BehaviorOptions(
                                                                    origin=admin_s3_origin,
                                                                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                                                                    function_associations=[cloudfront.FunctionAssociation(
                                                                        function=rewrite_function,
                                                                        event_type=cloudfront.FunctionEventType.VIEWER_REQUEST
                                                                    )]
                                                                ),
                                                                '/api/*': cloudfront.BehaviorOptions(
                                                                    origin=backend_origin,
                                                                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                                                                    origin_request_policy=origin_request_policy,
                                                                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                                                                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED
                                                                ),
                                                                '/adminapi/*': cloudfront.BehaviorOptions(
                                                                    origin=admin_origin,
                                                                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                                                                    origin_request_policy=cloudfront.OriginRequestPolicy.from_origin_request_policy_id(self, 'AllViewerExceptHostHeader', 'b689b0a8-53d0-40ab-baf2-68738e2966ac'),
                                                                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                                                                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED
                                                                )
                                                            }
                                              )
        
        _ = s3deploy.BucketDeployment(self, 'DeployFrontend',
                                      sources=[s3deploy.Source.asset('./twin/frontend/src')],
                                      destination_bucket=frontend_bucket,
                                      distribution=distribution,
                                      distribution_paths=['/'],
                                     )

        admin_frontend_deployment = s3deploy.BucketDeployment(self, 'DeployAdminFrontend',
                                      sources=[s3deploy.Source.asset('./twin/frontend/admin', bundling=BundlingOptions(
                                          bundling_file_access=BundlingFileAccess.VOLUME_COPY,
                                          image=DockerImage.from_registry('node:22'),
                                          user="root",
                                          command=[
                                              "sh", "-c",
                                              "npm ci && npm run build && cp -r build/* /asset-output/"
                                          ],
                                          working_directory="/asset-input",
                                      ))],
                                      destination_bucket=admin_frontend_bucket,
                                      destination_key_prefix='admin/',
                                      distribution=distribution,
                                      distribution_paths=['/admin/*']
                                     )

        env_js_content = cdk.Fn.join('', [
            'export const env={"PUBLIC_VITE_COGNITO_USER_POOL_ID":"',
            user_pool_id,
            '","PUBLIC_VITE_COGNITO_USER_POOL_CLIENT_ID":"',
            user_pool_client_id,
            '"}'
        ])

        admin_env_deployment = s3deploy.BucketDeployment(self, 'DeployAdminEnv',
                                      sources=[s3deploy.Source.data(
                                          '_app/env.js',
                                          env_js_content
                                      )],
                                      destination_bucket=admin_env_bucket,
                                      distribution=distribution,
                                      distribution_paths=['/_app/env.js'],
                                      content_type='application/javascript'
                                      )

        # Ensure env.js is deployed after the main admin frontend to avoid race condition
        admin_env_deployment.node.add_dependency(admin_frontend_deployment)

        
        self.domain_name = distribution.distribution_domain_name

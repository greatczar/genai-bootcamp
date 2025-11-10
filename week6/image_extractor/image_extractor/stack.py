import aws_cdk as cdk

from image_extractor.backend import Backend

class Extractor(cdk.Stack):
    def __init__(self, scope: cdk.App, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        backend = Backend(self, 'Backend')
        """
        frontend = Frontend(self, 'Frontend', backend_endpoint=backend.domain_name)

        _ = cdk.CfnOutput(
            self, 
            'FrontendURL',
            value=f"https://{frontend.domain_name}",
            description='Frontend UI URL'
        )
        """

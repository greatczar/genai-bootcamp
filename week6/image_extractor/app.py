#!/usr/bin/env python3
import os
import aws_cdk as cdk
from image_extractor.stack import Extractor


env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region='us-west-2')
app = cdk.App()
_ = Extractor(app, "ImageExtractorStack", env=env)

app.synth()

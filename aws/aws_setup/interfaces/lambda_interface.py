"""
Defines interfaces for all Amazon Lambda Functionalities.

Function Manager:
1. Create Function
2. Delete Function
3. List Functions

Deployment Manager:
1. Upload Code from ZIP
2. Upload Code from S3 Bucket
3. Generate Deployment Package
4. Publish New Version

Invocation:
1. Invoke Function
"""
from typing import Protocol, List, Dict, Optional, Any

class LambdaFunctionInterface(Protocol):
    def create_function(self,
                        func_name: str,
                        role_arn: str,
                        code: Dict[str, str],
                        handler: str,
                        timeout: int = 30) -> bool:
        raise NotImplementedError
    
    def delete_function(self, func_name: str) -> bool:
        raise NotImplementedError
    
    def list_functions(self) -> List[Dict[str, str]]:
        raise NotImplementedError

class LambdaDeploymentInterface(Protocol):
    def _generate_deployment_package(self, source_path: str) -> bytes:
        raise NotImplementedError
    
    def package_from_local_path(self, source_path: str) -> Dict[str, bytes]:
        raise NotImplementedError
    
    def upload_to_s3(self,
                     zip_bytes: bytes,
                     bucket: str,
                     key: str) -> bool:
        raise NotImplementedError
    
    def publish_new_version(self,
                            function_name: str,
                            description: Optional[str] = None) -> str:
        raise NotImplementedError
    
class LambdaInvocationInterface(Protocol):
    def invoke_function(self,
                        func_name: str,
                        payload: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError
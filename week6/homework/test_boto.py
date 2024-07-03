import boto3

# Configurar el cliente de S3 para apuntar a LocalStack
s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

# Listar los buckets
response = s3.list_buckets()

# Imprimir los nombres de los buckets
for bucket in response['Buckets']:
    print(bucket['Name'])
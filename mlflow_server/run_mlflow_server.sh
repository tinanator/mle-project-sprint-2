export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

# mlflow server \
#   --backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME\
#     --default-artifact-root s3://$AWS_BUCKET_NAME \
#     --no-serve-artifacts

mlflow server \
  --backend-store-uri postgresql://mle_20240328_6bcf522120:0c5b03147fbf49b585770e53719feac4@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20240328_6bcf522120\
    --default-artifact-root s3-student-mle-20240328-6bcf522120 \
    --no-serve-artifacts
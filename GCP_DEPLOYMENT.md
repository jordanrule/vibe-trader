# GCP Deployment Guide for Trading System

This guide provides step-by-step instructions to deploy the trading system to Google Cloud Platform using Cloud Functions, Cloud Storage, and Secret Manager.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed (for building container)
- Python 3.11+ installed locally

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud Function │    │  Cloud Storage  │    │ Secret Manager  │
│   (main_cloud.py)│◄──►│ (State files &  │    │ (API Keys)      │
│                  │    │   logs)         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│  External APIs   │
│  (Kraken, OpenAI,│
│   Telegram)      │
└─────────────────┘
```

## Step 1: Set Up GCP Project

1. **Create a new GCP project** (or use existing):
```bash
gcloud projects create your-trading-project
gcloud config set project your-trading-project
```

2. **Enable required APIs**:
```bash
# Enable Cloud Functions API
gcloud services enable cloudfunctions.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Enable Cloud Build API (for building containers)
gcloud services enable cloudbuild.googleapis.com
```

## Step 2: Create Cloud Storage Bucket

1. **Create a storage bucket** for state files and logs:
```bash
BUCKET_NAME="your-trading-project-storage"
gcloud storage buckets create gs://$BUCKET_NAME --location=us-central1
```

2. **Set bucket permissions** (optional - for easier access):
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
    --member="serviceAccount:your-project-number-compute@developer.gserviceaccount.com" \
    --role="roles/storage.admin"
```

## Step 3: Create Secrets in Secret Manager

Create secrets for all API keys and configuration:

1. **Kraken API Key**:
```bash
echo -n "your_kraken_api_key_here" | gcloud secrets create KRAKEN_API_KEY --data-file=-
```

2. **Kraken API Secret**:
```bash
echo -n "your_kraken_api_secret_here" | gcloud secrets create KRAKEN_API_SECRET --data-file=-
```

3. **OpenAI API Key**:
```bash
echo -n "your_openai_api_key_here" | gcloud secrets create OPENAI_API_KEY --data-file=-
```

4. **Telegram Bot Token**:
```bash
echo -n "your_telegram_bot_token_here" | gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=-
```

5. **GCP Configuration**:
```bash
# Project ID
echo -n "your-trading-project" | gcloud secrets create GCP_PROJECT_ID --data-file=-

# GCS Bucket Name
echo -n "your-trading-project-storage" | gcloud secrets create GCS_BUCKET_NAME --data-file=-
```

6. **Trading Configuration** (optional - defaults will be used if not set):
```bash
echo -n "true" | gcloud secrets create LIVE_MODE --data-file=-
echo -n "6" | gcloud secrets create MAX_TRADE_LIFETIME_HOURS --data-file=-
echo -n "20.0" | gcloud secrets create STOP_LOSS_PERCENTAGE --data-file=-
echo -n "INFO" | gcloud secrets create LOG_LEVEL --data-file=-
```

## Step 4: Build and Push Docker Container

1. **Build the Docker container**:
```bash
# Make sure you're in the project root directory
cd /path/to/vibe-trader

# Build the container
docker build -t gcr.io/your-trading-project/trading-system:latest .
```

2. **Push to Google Container Registry**:
```bash
# Authenticate Docker with GCR
gcloud auth configure-docker

# Push the container
docker push gcr.io/your-trading-project/trading-system:latest
```

## Step 5: Deploy Cloud Function

1. **Create the Cloud Function**:
```bash
gcloud functions deploy trading-system \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=. \
    --entry-point=cloud_function_handler \
    --trigger-http \
    --allow-unauthenticated \
    --memory=1GB \
    --timeout=600s \
    --set-env-vars="CLOUD_MODE=true,GCP_PROJECT_ID=your-trading-project" \
    --docker-registry=artifact-registry
```

2. **Alternative: Deploy with container** (recommended):
```bash
gcloud functions deploy trading-system \
    --gen2 \
    --region=us-central1 \
    --docker-repository=gcr.io/your-trading-project/trading-system \
    --entry-point=cloud_function_handler \
    --trigger-http \
    --allow-unauthenticated \
    --memory=2GB \
    --timeout=600s \
    --set-env-vars="CLOUD_MODE=true,GCP_PROJECT_ID=your-trading-project"
```

## Step 6: Set Up Cloud Scheduler (for Hourly Execution)

1. **Enable Cloud Scheduler API**:
```bash
gcloud services enable cloudscheduler.googleapis.com
```

2. **Create a Cloud Scheduler job**:
```bash
gcloud scheduler jobs create http trading-cycle-hourly \
    --schedule="0 * * * *" \
    --uri="https://us-central1-your-trading-project.cloudfunctions.net/trading-system" \
    --http-method=POST \
    --oidc-service-account-email="your-service-account@your-trading-project.iam.gserviceaccount.com"
```

## Step 7: Test the Deployment

1. **Test the Cloud Function manually**:
```bash
# Get the function URL
gcloud functions describe trading-system --region=us-central1 --format="value(url)"

# Test with curl
curl -X POST https://us-central1-your-trading-project.cloudfunctions.net/trading-system
```

2. **Check logs**:
```bash
# View Cloud Function logs
gcloud functions logs read trading-system --region=us-central1 --limit=50
```

3. **Verify Cloud Storage files**:
```bash
# List files in your bucket
gcloud storage ls gs://your-trading-project-storage/**

# View state files
gcloud storage cat gs://your-trading-project-storage/state_opportunities.json
```

## Step 8: Monitoring and Maintenance

### View Logs
```bash
# Recent function logs
gcloud functions logs read trading-system --region=us-central1 --limit=20

# Logs with specific time range
gcloud functions logs read trading-system \
    --region=us-central1 \
    --start-time="2024-01-01T00:00:00Z" \
    --end-time="2024-01-02T00:00:00Z"
```

### Monitor Function Performance
```bash
# View function metrics
gcloud functions describe trading-system --region=us-central1

# View execution times and errors
gcloud logging read "resource.type=cloud_function AND resource.labels.function_name=trading-system" --limit=10
```

### Update the Function
```bash
# Update with new code
gcloud functions deploy trading-system \
    --source=. \
    --region=us-central1
```

## Configuration Options

### Environment Variables
- `CLOUD_MODE=true`: Enables cloud mode (automatically detected)
- `GCP_PROJECT_ID`: Your GCP project ID
- `LIVE_MODE=true`: Enable live trading (default: false)
- `MAX_TRADE_LIFETIME_HOURS=6`: Trade lifetime in hours
- `STOP_LOSS_PERCENTAGE=20.0`: Stop loss percentage
- `LOG_LEVEL=INFO`: Logging level

### Secret Manager Secrets
- `KRAKEN_API_KEY`: Kraken API key
- `KRAKEN_API_SECRET`: Kraken API secret
- `OPENAI_API_KEY`: OpenAI API key
- `TELEGRAM_BOT_TOKEN`: Telegram bot token
- `GCS_BUCKET_NAME`: Cloud Storage bucket name
- `GCP_PROJECT_ID`: GCP project ID

## Troubleshooting

### Common Issues

1. **Function times out**:
   - Increase memory allocation: `--memory=2GB`
   - Increase timeout: `--timeout=900s`

2. **Storage permission errors**:
   ```bash
   gcloud projects add-iam-policy-binding your-trading-project \
       --member="serviceAccount:your-project-number@appspot.gserviceaccount.com" \
       --role="roles/storage.admin"
   ```

3. **Secret access errors**:
   ```bash
   gcloud secrets add-iam-policy-binding SECRET_NAME \
       --member="serviceAccount:your-project-number@appspot.gserviceaccount.com" \
       --role="roles/secretmanager.secretAccessor"
   ```

4. **Container build fails**:
   - Check Docker build logs
   - Ensure all dependencies are in requirements.txt
   - Verify Python version compatibility

### Local Testing

Test locally before deploying:
```bash
# Set local environment variables
export KRAKEN_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
# ... other env vars

# Test locally
python main_cloud.py
```

### Backup and Recovery

1. **Backup state files**:
```bash
# Download all state files
gcloud storage cp gs://your-bucket/state_* ./backup/

# Download logs
gcloud storage cp gs://your-bucket/logs/* ./backup/logs/
```

2. **Restore from backup**:
```bash
# Upload state files
gcloud storage cp ./backup/state_* gs://your-bucket/

# Upload logs
gcloud storage cp ./backup/logs/* gs://your-bucket/logs/
```

## Cost Optimization

1. **Function Memory**: Start with 1GB, increase if needed
2. **Timeout**: Keep under 10 minutes to control costs
3. **Storage Class**: Use Standard storage for active data
4. **Scheduler**: Hourly execution is reasonable for trading system

## Security Best Practices

1. **Least Privilege**: Grant minimal IAM permissions
2. **Secret Rotation**: Regularly rotate API keys and secrets
3. **Network Security**: Use VPC if accessing internal resources
4. **Audit Logging**: Enable Cloud Audit Logs for compliance

---

## Summary

✅ **Cloud Function**: Deploys trading system as serverless function
✅ **Cloud Storage**: Handles all state files and logs
✅ **Secret Manager**: Securely stores API keys and configuration
✅ **Cloud Scheduler**: Automates hourly execution
✅ **Environment Detection**: Seamlessly works in both local and cloud modes

The system is now ready for production deployment on Google Cloud Platform! 🚀

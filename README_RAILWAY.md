# Deploy on Railway

## 🚀 Railway Deployment Guide

Railway is better for ML apps because it has more memory and handles PyTorch better than Render.

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Create a new project

### Step 2: Connect Repository
1. Click "Deploy from GitHub repo"
2. Select your repository: `ShubhamSoni09/Deep-Image-Refinement-using-GANs`
3. Railway will auto-detect Python

### Step 3: Configure Settings
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --timeout 300 --workers 1`
- **Environment**: Python 3.11

### Step 4: Deploy
1. Click "Deploy Now"
2. Wait for build to complete
3. Your app will be live at the provided URL

## 🎯 Why Railway is Better:
- ✅ **More Memory**: 512MB vs 256MB on Render
- ✅ **Better ML Support**: Handles PyTorch models better
- ✅ **Longer Timeouts**: 300s vs 30s on Render
- ✅ **Auto-restart**: Handles crashes gracefully

## 🔧 Configuration Files:
- `railway.json`: Railway-specific settings
- `runtime.txt`: Python version specification
- `requirements.txt`: Dependencies with specific versions

## 🚨 If Still Having Issues:
1. **Reduce image size** to 128px max
2. **Use only one GAN** at a time
3. **Upgrade to paid plan** for more resources 
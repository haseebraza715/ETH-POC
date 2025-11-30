# Streamlit Cloud Deployment Guide

## Prerequisites

1. Your code is pushed to GitHub: `https://github.com/haseebraza715/ETH-POC.git`
2. You have a Streamlit Cloud account (free at https://streamlit.io/cloud)

## Deployment Steps

### 1. Go to Streamlit Cloud

Visit: https://share.streamlit.io/

### 2. Sign in with GitHub

Click "Sign in" and authorize Streamlit Cloud to access your GitHub repositories.

### 3. Deploy Your App

1. Click "New app"
2. Select your repository: `haseebraza715/ETH-POC`
3. Select branch: `main`
4. Main file path: `app.py`
5. Click "Deploy!"

### 4. Configure Environment Variables (Optional but Recommended)

After deployment, go to your app settings and add these secrets:

**For OpenRouter (recommended for free tier):**
- `OPENROUTER_API_KEY` - Your OpenRouter API key

**OR for OpenAI:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Optional OpenRouter settings:**
- `OPENROUTER_MODEL` - Model name (default: `mistralai/mistral-small-3.2-24b-instruct:free`)
- `OPENROUTER_SITE_URL` - Your site URL
- `OPENROUTER_SITE_NAME` - Your site name

> **Note:** Without API keys, the app will still work but use rule-based fallback extraction (less accurate).

### 5. Access Your Deployed App

Once deployed, Streamlit Cloud will provide you with a URL like:
`https://your-app-name.streamlit.app`

## Troubleshooting

- **Build fails?** Check that `requirements.txt` includes all dependencies
- **App crashes?** Check the logs in Streamlit Cloud dashboard
- **API errors?** Verify your environment variables are set correctly

## Updating Your App

Simply push changes to your `main` branch on GitHub, and Streamlit Cloud will automatically redeploy!


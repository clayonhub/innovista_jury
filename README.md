# Tata Innovista Jury Matching System

An AI-powered application to semantically match faculty expertise to project requirements.

## Cloud Deployment Guide (Streamlit Community Cloud)

This app is ready to be deployed to the web via **Streamlit Community Cloud** for free.

### Step 1: Push code to GitHub
You need to get this folder onto a public (or private) GitHub repository first. Open your terminal in this folder and run:
```bash
git add .
git commit -m "Initial commit for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
*Note: Make sure your `faculty_master_list.csv`, `faculty_minilm_embeddings.npy`, and `sessions.json` files are successfully pushed. The `.gitignore` in this repo is specially configured to ensure they are uploaded.*

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with the same GitHub account.
2. Click **New app**.
3. Fill in the deployment details:
   - **Repository:** Select the repository you just created (e.g., `YOUR_USERNAME/YOUR_REPO_NAME`)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Optional) You can choose a custom URL here like `tata-innovista-matching`.
4. Click **Deploy!**

### Step 3: Wait for Setup
The first time it boots up, Streamlit Cloud will download the Machine Learning model (`all-MiniLM-L6-v2`) in the background. This usually takes about 30 seconds. Afterwards, the app will instantly boot and your workspaces will be available online.

---
### Local Usage
If you want to run this locally rather than in the cloud:
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

# Medical Imaging CV - GitHub Setup Guide

Create and submit your medical imaging project on GitHub.

---

## Step 1: Create GitHub Account (if needed)

1. Go to https://github.com/signup
2. Sign up with your email
3. Create a username
4. Verify your email

---

## Step 2: Create New Repository

1. Go to https://github.com/new
2. **Repository name**: `medical-imaging-cv`
3. **Description**: "AI-Powered Chest X-ray Disease Detection using Deep Learning"
4. **Public**: ✓ Check (required for submission)
5. **Initialize**: Do NOT check any boxes
6. Click **"Create repository"**

You'll see a page with setup instructions. Copy the HTTPS URL (looks like `https://github.com/yourusername/medical-imaging-cv.git`)

---

## Step 3: Push Code to GitHub

Open terminal in your `medical-imaging-cv/` folder:

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Medical Image Analysis System with CNN"

# Add remote (replace URL with yours)
git remote add origin https://github.com/yourusername/medical-imaging-cv.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**That's it!** Your code is now on GitHub.

---

## Step 4: Verify on GitHub

1. Go to `https://github.com/yourusername/medical-imaging-cv`
2. You should see all your files:
   - medical_imaging_analyzer.py
   - app.py
   - templates/index.html
   - test_medical_imaging.py
   - README.md
   - PROJECT_REPORT.md
   - requirements.txt
   - .gitignore

3. Click on **README.md** - it should display with formatting
4. Check **PROJECT_REPORT.md** displays properly

---

## Step 5: Verify Public Access

Open an **incognito/private browser window** and go to your repo URL:
`https://github.com/yourusername/medical-imaging-cv`

**Must be able to see** all files without logging in!

---

## Using GitHub Desktop (GUI Alternative)

If command line is unfamiliar:

1. Download https://desktop.github.com/
2. Sign in with GitHub account
3. Click "Create a New Repository"
4. **Name**: `medical-imaging-cv`
5. **Local Path**: Where your code is
6. Click "Create Repository"
7. Click "Publish repository"
8. Verify files show on github.com

---

## Git Workflow for Later Updates

After initial setup, making updates is easy:

```bash
# See what changed
git status

# Add changed files
git add .

# Commit with message
git commit -m "Fix: Better CLAHE parameters for medical images"

# Push to GitHub
git push
```

---

## Troubleshooting

### "fatal: not a git repository"
- Run `git init` first

### "Permission denied (publickey)"
- Use HTTPS URL instead of SSH
- Or set up SSH keys (GitHub docs)

### "Repository not found"
- Check URL is correct
- Verify you created the repo on GitHub

### Files don't appear on GitHub
- Run `git push` to upload
- Refresh GitHub website
- Wait a few seconds for update

### Files uploaded but not showing
- Check .gitignore isn't excluding them
- Ensure you did `git add .`

---

## Verification Checklist

Before submitting, verify:

- [ ] Repository is public (no login needed to view)
- [ ] All files present on GitHub:
  - [ ] medical_imaging_analyzer.py
  - [ ] app.py
  - [ ] templates/index.html
  - [ ] test_medical_imaging.py
  - [ ] requirements.txt
  - [ ] README.md
  - [ ] PROJECT_REPORT.md
  - [ ] .gitignore
- [ ] README.md displays with formatting
- [ ] PROJECT_REPORT.md is readable
- [ ] No __pycache__ or .pyc files
- [ ] No venv/ or uploads/ folders
- [ ] At least 2-3 commits in history

---

## GitHub Tips

### Good Commit Messages
```bash
git commit -m "Add: CLAHE enhancement for medical images"
git commit -m "Fix: Model weight initialization"
git commit -m "Doc: Add API reference to README"
```

### See Commit History
Click on the commit count on GitHub homepage.

### Edit Files on GitHub
Click on file → Click pencil icon → Edit → Commit

### Delete/Rename Files on GitHub
Files view → Click file → Click trash or ... menu

---

## For Submission

**Your GitHub URL**: `https://github.com/yourusername/medical-imaging-cv`

Copy this URL and use for:
1. Submitting to VITyarthi
2. Course assignment portal
3. Any documentation

---

## Next Steps

1. ✅ Create GitHub repository (done!)
2. ✅ Push your code to GitHub (done!)
3. → Go to `SUBMISSION_CHECKLIST.md` for final submission
4. → Submit URL to VITyarthi platform

---

**Your code is now safely version-controlled and publicly available!** 🎉

How to publish this local clone to your GitHub account

This repo is a cloned copy of `gpu-mode/reference-kernels`. If you want to push your local copy to your own GitHub account for safekeeping or later work, follow the safe steps below.

Recommended workflow (manual, safe)

1) Create an empty repository on GitHub (via the website) or use the GitHub CLI:

   gh repo create youruser/reference-kernels --private --confirm

2) From your local repo root, add your repo as the origin remote and push:

   # replace with your remote URL (SSH or HTTPS)
   ./scripts/publish_to_github.sh git@github.com:youruser/reference-kernels.git main

3) Verify on GitHub the code is present.

Notes and alternatives
- If you want to keep the original remote, you can add your remote under a different name, e.g. `myorigin`:

   git remote add myorigin git@github.com:youruser/reference-kernels.git
   git push -u myorigin main

- If you prefer HTTPS and want to avoid SSH keys:

   ./scripts/publish_to_github.sh https://github.com/youruser/reference-kernels.git main

- To avoid pushing large or sensitive files accidentally, inspect `git status` and `git ls-files` before pushing.

Automation (GH CLI)
- The GH CLI can create a repo and set it as remote automatically:

   gh repo create youruser/reference-kernels --private --source=. --remote=origin --push

Security
- This script does not send anything to the network by itself; it only runs git commands.
- You will be prompted or need to have credentials configured (SSH keys or gh auth) to push.

If you want, I can add a tiny GitHub Actions workflow to automatically push changes to a repo you specify (requires storing a PAT) — say the word and I will add it.

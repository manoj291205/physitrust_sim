#!/usr/bin/env bash
# Pushes this folder to https://github.com/manoj291205/physitrust_sim
# under YOUR name only. Run from the repository folder:
#     bash scripts/push_to_github.sh
set -e
REMOTE="https://github.com/manoj291205/physitrust_sim.git"

read -rp "Your full name for commits (e.g. Manojkumar Ashokkumar): " NAME
read -rp "Your GitHub email (the one on your GitHub account): " EMAIL

[ -d .git ] || git init -b main
git checkout -B main
git config user.name  "$NAME"
git config user.email "$EMAIL"
git add .
git commit -m "Initial commit: PhysiTrust-FL simulation, results and paper" || true

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"

# If the GitHub repo was created with a README or licence, merge it first
# and keep this folder's version of any file that exists in both.
if git ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
    git fetch origin main
    git merge origin/main --allow-unrelated-histories -X ours \
        -m "Merge files created on GitHub"
fi

git push -u origin main
echo "Done: https://github.com/manoj291205/physitrust_sim"

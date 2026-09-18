#!/usr/bin/env bash
# Open the awesome-dsh-plugin PR for dsh-plugin-c2c.
#
# The fork branch is already pushed; the repo's CI refuses entries whose
# repository is younger than 1 day, so this is meant to run once that window
# has passed. It is safe to re-run: gh pr create fails if a PR already exists,
# and the age gate clears on its own.
set -euo pipefail

OWNER="Alyosha28"
REPO="dsh-plugin-c2c"
UPSTREAM="awesome-dsh-plugin/awesome-dsh-plugin"
HEAD_BRANCH="add-dsh-plugin-c2c"
BODY="$(dirname "$0")/PR_BODY.md"

echo "==> checking repo age"
created=$(gh api "repos/$OWNER/$REPO" --jq .created_at)
python3 - "$created" <<'PY'
import sys
from datetime import datetime, timezone
created = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
age = (datetime.now(timezone.utc) - created).total_seconds() / 3600
print(f"    {sys.argv[1]}  ({age:.2f}h old)")
if age < 24:
    raise SystemExit(
        f"    still short of the 1-day gate; retry in {24 - age:.1f}h. "
        "Pushing now would just fail CI."
    )
PY

echo "==> confirming the branch is pushed"
git ls-remote --exit-code --heads "https://github.com/$OWNER/awesome-dsh-plugin.git" "$HEAD_BRANCH" >/dev/null
echo "    $HEAD_BRANCH present"

echo "==> opening the PR"
gh pr create \
  --repo "$UPSTREAM" \
  --head "$OWNER:$HEAD_BRANCH" \
  --base main \
  --title "Add $OWNER/$REPO" \
  --body-file "$BODY"

echo "==> done"

#!/usr/bin/env bash
# Relocate the C2C work out of ~/factor_digging (an unrelated quant project)
# into ~/DSH, then re-point the DSH profile at the new locations.
#
# Idempotent and safe to re-run. Every step checks its own preconditions.
#
#   bash relocate-to-DSH.sh --dry-run     # show what would happen
#   bash relocate-to-DSH.sh               # do it
#
# What it does NOT touch:
#   ~/DSH/README.md  — a separate session's workspace troubleshooting note.
# Assumptions:
#   ~/DSH/dsh-plugin-c2c is a stale earlier copy of this plugin (verified: it has
#   no unique files). It is moved aside to dsh-plugin-c2c.old-<timestamp> rather
#   than deleted, so nothing is destroyed.

set -euo pipefail

SRC_ROOT="$HOME/factor_digging"
DST_ROOT="$HOME/DSH"
PLUGIN_SRC="$SRC_ROOT/dsh-plugin-c2c"
PLUGIN_DST="$DST_ROOT/dsh-plugin-c2c"
C2C_SRC="$SRC_ROOT/c2c"
C2C_DST="$DST_ROOT/c2c"
PROFILE_DIR="$HOME/.dsh/profiles/web"
STAMP="$(date +%Y%m%d-%H%M%S)"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say()  { printf '  %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
run()  { if [ "$DRY" = 1 ]; then say "would: $*"; else "$@"; fi; }

step "preconditions"
[ -d "$SRC_ROOT" ] || { echo "missing $SRC_ROOT"; exit 1; }
[ -d "$DST_ROOT" ] || { echo "missing $DST_ROOT"; exit 1; }
[ -d "$PLUGIN_SRC" ] || { echo "missing $PLUGIN_SRC"; exit 1; }
say "source : $SRC_ROOT"
say "target : $DST_ROOT"
say "mode   : $([ "$DRY" = 1 ] && echo 'DRY RUN' || echo 'APPLY')"

# Guard: never clobber a newer copy that happens to be at the destination.
if [ -d "$PLUGIN_DST/.git" ]; then
  echo "ERROR: $PLUGIN_DST is already a git checkout — refusing to touch it."
  echo "       Move it aside yourself and re-run."
  exit 1
fi

step "1/6 preserve the stale copy already at the destination"
if [ -d "$PLUGIN_DST" ]; then
  say "found existing $PLUGIN_DST (stale snapshot)"
  run mv "$PLUGIN_DST" "$PLUGIN_DST.old-$STAMP"
  say "kept as dsh-plugin-c2c.old-$STAMP"
else
  say "nothing at $PLUGIN_DST — skipping"
fi

step "2/6 move the plugin checkout (with its .git)"
if [ -d "$PLUGIN_SRC" ]; then
  run mv "$PLUGIN_SRC" "$PLUGIN_DST"
  say "moved -> $PLUGIN_DST"
else
  say "already moved"
fi

step "3/6 move the C2C runtime (venv + 3.2G weight cache)"
if [ -d "$C2C_SRC" ] && [ ! -d "$C2C_DST" ]; then
  run mv "$C2C_SRC" "$C2C_DST"
  say "moved -> $C2C_DST"
elif [ -d "$C2C_DST" ]; then
  say "$C2C_DST already exists — skipping (resolve by hand if both exist)"
else
  say "no $C2C_SRC to move"
fi

step "4/6 repoint the virtualenv's baked-in absolute paths"
# In dry-run the directory has not moved yet, so inspect the source location.
if [ "$DRY" = 1 ]; then VENV="$C2C_SRC/.venv"; else VENV="$C2C_DST/.venv"; fi
if [ -d "$VENV" ]; then
  if [ "$DRY" = 1 ]; then
    say "would rewrite $C2C_SRC -> $C2C_DST in $VENV/bin/* and pyvenv.cfg"
  else
    # Shebangs in bin/* plus 'command =' in pyvenv.cfg carry the old absolute
    # path. 'home =' points at the base interpreter and is left alone.
    fixed=$(grep -rl "$C2C_SRC" "$VENV/bin" "$VENV/pyvenv.cfg" 2>/dev/null | wc -l | tr -d ' ')
    grep -rl "$C2C_SRC" "$VENV/bin" "$VENV/pyvenv.cfg" 2>/dev/null \
      | while read -r f; do
          sed -i '' "s|$C2C_SRC|$C2C_DST|g" "$f"
        done
    say "rewrote absolute paths in $fixed file(s)"
    if "$VENV/bin/python" -c "import sys" 2>/dev/null; then
      say "venv interpreter responds: $("$VENV/bin/python" -V 2>&1)"
    else
      say "WARNING: venv interpreter did not respond; recreate with:"
      say "  python $PLUGIN_DST/python/dsh_c2c_setup.py --root $C2C_DST --recreate"
    fi
  fi
else
  say "no venv at $VENV"
fi

step "5/6 re-point the DSH profile config and plugin symlink"
# The profile holds a symlink to the OLD checkout path. Moving the checkout
# leaves it dangling, and pnpm may not replace a broken link cleanly, so drop
# it first. It is recreated by the `dsh plugin add` below.
LINK="$PROFILE_DIR/node_modules/dsh-plugin-c2c"
if [ -L "$LINK" ]; then
  target="$(readlink "$LINK")"
  case "$target" in
    *"/factor_digging/dsh-plugin-c2c")
      run rm "$LINK"
      say "removed dangling profile symlink (was $target)"
      ;;
    *) say "profile symlink points elsewhere ($target) — leaving it" ;;
  esac
fi

PATCH="$PROFILE_DIR/cordis.patch.yml"
if [ -f "$PATCH" ]; then
  if [ "$DRY" = 1 ]; then
    say "would set repoRoot/pythonPath in $PATCH"
  else
    cp "$PATCH" "$PATCH.bak-$STAMP"
    if grep -q '^ *repoRoot:' "$PATCH"; then
      sed -i '' "s|^\( *\)repoRoot:.*|\1repoRoot: $C2C_DST|" "$PATCH"
    else
      printf -- "- id: c2c\n  config:\n    repoRoot: %s\n" "$C2C_DST" >> "$PATCH"
    fi
    if grep -q '^ *pythonPath:' "$PATCH"; then
      sed -i '' "s|^\( *\)pythonPath:.*|\1pythonPath: $VENV/bin/python|" "$PATCH"
    fi
    say "updated $PATCH (backup at cordis.patch.yml.bak-$STAMP)"
  fi
else
  say "no $PATCH — the plugin's auto-detection will find $C2C_DST anyway"
fi

if [ "$DRY" = 0 ]; then
  say "reinstalling the plugin so the symlink points at the new path"
  dsh plugin --profile web add "$PLUGIN_DST" >/dev/null 2>&1 \
    && say "symlink refreshed" \
    || say "WARNING: 'dsh plugin add' failed — run it by hand"
else
  say "would run: dsh plugin --profile web add $PLUGIN_DST"
fi

step "6/6 verify"
if [ "$DRY" = 0 ]; then
  say "plugin files : $(find "$PLUGIN_DST" -maxdepth 1 -type f | wc -l | tr -d ' ')"
  say "git remote   : $(git -C "$PLUGIN_DST" remote get-url origin 2>/dev/null || echo none)"
  say "git status   : $(git -C "$PLUGIN_DST" status --porcelain | wc -l | tr -d ' ') changed"
  say "rosetta      : $([ -d "$C2C_DST/rosetta" ] && echo present || echo MISSING)"
  say "weights      : $(du -sh "$C2C_DST/models" 2>/dev/null | cut -f1 || echo none)"
  say "venv python  : $([ -x "$VENV/bin/python" ] && echo ok || echo MISSING)"

  cd "$PROFILE_DIR"
  node --input-type=module -e "
    const m = await import('dsh-plugin-c2c');
    const reg = new Map();
    m.apply({tools:{register:d=>reg.set(d.name,d)},systemPrompt:{section(){}},logger:{info(){}},on(){}},{});
    const s = await reg.get('c2c_status').execute({start:false}, {});
    console.log('  tools        :', reg.size);
    console.log('  repoRoot     :', s.repoRoot);
    console.log('  python       :', s.python);
  " 2>&1 | sed 's/^/  /' || say "plugin load check failed"
else
  say "would verify plugin load and auto-detected paths"
fi

if [ "$DRY" = 1 ]; then
  printf '\nDRY RUN complete — nothing was changed.\n'
else
  printf '\nDone. Restart the DSH session to pick up the new paths.\n'
fi

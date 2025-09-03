#!/bin/bash
# Quick HF Clone Debug Script

echo "🔍 QUICK HF CLONE DIAGNOSTIC"
echo "============================"

echo "1. Repository Size:"
echo "   Total: $(du -sh . | cut -f1)"
echo "   Git:   $(du -sh .git | cut -f1)"

echo ""
echo "2. Submodule Check:"
if [ -f .gitmodules ]; then
    echo "   ✅ .gitmodules exists"
    echo "   Content:"
    cat .gitmodules | sed 's/^/      /'
    echo "   Status:"
    git submodule status 2>&1 | sed 's/^/      /'
else
    echo "   ❌ No .gitmodules"
fi

echo ""
echo "3. Large Files (>5MB):"
find . -size +5M -type f 2>/dev/null | head -5 | sed 's/^/   /'

echo ""
echo "4. Git Integrity:"
git fsck --no-progress 2>&1 | head -3 | sed 's/^/   /'

echo ""
echo "5. Clone Test:"
TEMP_DIR=$(mktemp -d)
echo "   Testing clone to: $TEMP_DIR"
if git clone --depth 1 . "$TEMP_DIR/test" &>/dev/null; then
    echo "   ✅ Shallow clone successful"
    echo "   Size: $(du -sh "$TEMP_DIR/test" | cut -f1)"
else
    echo "   ❌ Shallow clone failed"
fi
rm -rf "$TEMP_DIR"

echo ""
echo "6. Submodule URL Test:"
if curl -s -o /dev/null -w "%{http_code}" https://github.com/yakymchukluka-afk/stylegan-v | grep -q "200"; then
    echo "   ✅ Submodule URL accessible"
else
    echo "   ❌ Submodule URL not accessible"
fi

echo ""
echo "🎯 DIAGNOSIS COMPLETE"
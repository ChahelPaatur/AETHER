#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: bash bump_version.sh 3.1.0"
    exit 1
fi

VERSION=$1
echo "Releasing AETHER v$VERSION..."

# Update VERSION file
echo "$VERSION" > VERSION

# Update pyproject.toml
sed -i '' "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Update __init__.py
sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" aether/__init__.py

# Commit and tag
git add VERSION pyproject.toml aether/__init__.py
git commit -m "Release v$VERSION"
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo "Released v$VERSION — GitHub Actions will publish to PyPI automatically"
echo "Watch: https://github.com/ChahelPaatur/AETHER/actions"

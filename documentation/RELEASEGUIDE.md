In order to publish a new version to PyPI, please follow the following steps:

## 1. Add your changes to the changelog

Please note any major changes in the CHANGELOG.md.

## 2. Update Version with bump2version

To update the version number of PyRates, please use [bump2version](https://github.com/c4urself/bump2version) to ensure
consistency.
The version number follows the pattern `<major>.<minor>.<patch>[-<release><build>]`, where the release
suffix is optional and cycles through `dev → rc → (stable)`.

`bump2version` auto-creates a commit for every version change.

**Typical release workflow for a new patch version:**

```
# 1. Start development build
bump2version patch                            # e.g. 1.0.10 → 1.1.0-dev1

# 2. Iterate on the development build as needed
bump2version build                            # e.g. 1.1.0-dev1 → 1.1.0-dev2

# 3. Promote to release candidate for final testing
bump2version --new-version 1.1.0-rc1 patch   # e.g. 1.1.0-dev2 → 1.1.0-rc1

# 4. Mark as stable once tests pass
bump2version --new-version 1.1.0 patch       # e.g. 1.1.0-rc1 → 1.1.0
```

Use `--allow-dirty` if you have uncommitted local changes that should be included in the bump commit
(e.g. an edited `.bumpversion.cfg`):
```
bump2version --new-version 1.1.0-rc1 --allow-dirty patch
```

## 3. Make a release using github.com

The PyRates repository is currently configured such that a release on github.com is automatically pushed to PyPI. 
For more information on creating a release on github.com, please look 
[here](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository).

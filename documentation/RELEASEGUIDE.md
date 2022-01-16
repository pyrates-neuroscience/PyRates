In order to publish a new version to PyPI, please follow the following steps:

## 1. Add your changes to the changelog

Please note any major changes in the CHANGELOG.md.

## 2. Update Version with bump2version

To update the version number of PyRates, please use [bump2version](https://github.com/c4urself/bump2version) to ensure
consistency. 
The version number consists of the following components `<major>.<minor>.<patch>-<build>`. 

For example, to start the next "patch" version, type
```
bump2version patch  # updates "0.0.0" to "0.0.1-dev0"
```

This will increase the patch number and append a "-dev0" to indicate this is a development build. 
`bump2version` also auto-creates a commit, indicating in the commit message how the version was modified.  

The following command updates the build number of the development build (`-dev#`).
```
bump2version build  # updates "0.0.1-dev0" to "0.0.1-dev1"
```

To mark the current version as stable, type
```
bump2version release  # updates "0.0.1-dev1" to "0.0.1"
```

## 3. Make a release using github.com

The PyRates repository is currently configured such that a release on github.com is automatically pushed to PyPI. 
For more information on creating a release on github.com, please look 
[here](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository).

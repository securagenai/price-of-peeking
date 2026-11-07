# Releasing a new version

The Zenodo record ([10.5281/zenodo.22995998](https://doi.org/10.5281/zenodo.22995998); all versions: [10.5281/zenodo.22995997](https://doi.org/10.5281/zenodo.22995997)) is uploaded manually; it is not created automatically from GitHub releases.

1. Update the version and `date-released` in `CITATION.cff`, and `README.md`, `LICENSING.md` and `NOTICE` if their content changes.
2. Add every new file to `docs/LICENSING_MAP.json` with its licence scope, and remove deleted files.
3. Stage all changes (`git add`), then update and check `MANIFEST.sha256`:

   ```bash
   git ls-files | grep -v '^MANIFEST.sha256$' | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256
   sha256sum -c MANIFEST.sha256
   ```

4. Commit, then create a GitHub release with tag `vX.Y.Z`.
5. On Zenodo, open the record, choose **New version**, upload the release archive and update the version, description and related identifiers. Keep both licences: **Apache-2.0** for code and executable configurations and **CC BY 4.0** for documentation, aggregate results and figures. Publish and check that the record lists both licences.

If the GitHub–Zenodo integration is ever enabled for this repository, first add a `.zenodo.json` with a single licence identifier (Zenodo does not accept a list of licences there or in `CITATION.cff`) and add the second licence on Zenodo after each release.

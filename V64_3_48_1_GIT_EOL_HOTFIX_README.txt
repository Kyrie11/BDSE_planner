V64.3.48.1 Git EOL engineering hotfix

Algorithm: unchanged.
Root cause: three source-manifested text files were packaged with CRLF bytes while Git normalized them to LF during commit/checkout, causing sha256sum -c to report FAILED.

Changed source bytes only:
- bdse/nuplan_config/splitter/nuplan.yaml: CRLF -> LF
- bdse/tests/test_candidate_valid_repair.py: CRLF -> LF
- bdse/tests/test_nuplan_future_next_token_fallback.py: CRLF -> LF

Added:
- .gitattributes with eol=lf for exactly the three paths above.

Updated:
- V64_3_48_SOURCE_MANIFEST.sha256 (896 entries; includes .gitattributes)

Validation:
- source manifest: 896/896 PASS
- direct affected/V48 tests: 9/9 PASS
- V13->V48 targeted regression: 232/232 PASS
- simulated Git commit with core.autocrlf=true -> Linux clone with core.autocrlf=false -> source manifest: 896/896 PASS

When copying into an existing repository, include the hidden .gitattributes file. Prefer:
  rsync -a <fixed_dir>/ <repo_root>/
or:
  cp -a <fixed_dir>/. <repo_root>/
Do not use only "cp <fixed_dir>/* <repo_root>/" because shell * omits .gitattributes.

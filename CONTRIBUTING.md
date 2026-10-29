# Contributing

Describe the problem, the changed behaviour and the checks you ran. Run the frontend build, backend regression suite and shared-source hash check before proposing a change. A numerical change should include units, physical conventions and a reproducible small case, and explain which validation results need rerunning.

Keep translations in `src/translations.json` in sync. Changing language must preserve jobs and inputs. Do not rename exported CSV/JSON fields for translation.

The simulation uses bundled numerical primitives under `vendor/seaarray_core/`. Verify them with `scripts/sync_core.py`; use `--source PATH` to compare with the recorded upstream revision. Updating the numerical core requires reviewing source changes alongside the calling code and deliberately regenerating the manifest. The checker never silently overwrites the English documentation or portable configuration subset. Preserve failures and uncertainty limits in reproduction reports. Do not add raw datasets beyond the approved example files, generated outputs, virtual environments or local logs. Contributions are provided under the MIT license.

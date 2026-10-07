# Contributing

Describe the problem, the changed behaviour and the checks you ran. Run the frontend build, backend regression suite and shared-source hash check before proposing a change. A numerical change should include units, physical conventions and a reproducible small case, and explain which validation results need rerunning.

Keep translations in `src/translations.json` in sync. Changing language must preserve jobs and inputs. Do not rename exported CSV/JSON fields for translation.

The simulation uses bundled numerical primitives under `vendor/seaarray_core/`. Update them through `scripts/sync_core.py`, with the source changes reviewed alongside the calling code. Preserve failures and uncertainty limits in experiment reports. Do not add large raw datasets, generated outputs, virtual environments or local logs. Contributions are provided under the MIT license.

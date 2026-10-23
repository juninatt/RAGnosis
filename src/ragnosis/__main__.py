"""Allow ``python -m ragnosis``. Python only runs this file in that case, so no main guard."""

from ragnosis.cli import main

raise SystemExit(main())

"""warning filter setup — narrow and idempotent."""
import warnings

from operation_love import _warnings as ow


def test_configure_warnings_adds_two_filters_once():
    old_filters = list(warnings.filters)
    old_configured = ow._CONFIGURED
    try:
        warnings.filters[:] = []
        ow._CONFIGURED = False

        ow.configure_warnings()
        ow.configure_warnings()

        assert len(warnings.filters) == 2
        got = {
            (f[0], f[1].pattern, f[2], f[3].pattern if f[3] else "")
            for f in warnings.filters
        }
        assert got == {
            ("ignore", r"pkg_resources is deprecated as an API", UserWarning, r"clip(\.|$)"),
            ("ignore", r"`estimate` is deprecated", FutureWarning,
             r"insightface\.utils\.face_align$"),
        }
    finally:
        warnings.filters[:] = old_filters
        ow._CONFIGURED = old_configured

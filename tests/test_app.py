import pytest
from streamlit.testing.v1 import AppTest

import ringvax.app


@pytest.mark.filterwarnings(
    r"ignore:\s+Deprecated since `altair=5.5.0`. Use altair.theme instead."
)
def test_app():
    # Cf. https://docs.streamlit.io/develop/api-reference/app-testing
    at = AppTest.from_file("ringvax/app.py", default_timeout=10.0).run()
    assert not at.exception


def test_get_commit():
    """
    get_commit should return a string

    This presumes that pytest is always run in the context of a repo,
    which seems like a fair assumption. The alternative is mocking, which
    seems like overkill
    """
    commit = ringvax.app.get_commit()
    assert isinstance(commit, str)

from streamlit.testing.v1 import AppTest


def test_app():
    # Cf. https://docs.streamlit.io/develop/api-reference/app-testing
    at = AppTest.from_file("ringvax/app.py", default_timeout=10.0).run()
    assert not at.exception

from streamlit.testing.v1 import AppTest


def test_app():
    # Cf. https://docs.streamlit.io/develop/api-reference/app-testing
    at = AppTest.from_file("ringvax/app.py").run()
    assert not at.exception

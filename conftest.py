"""Pytest configuration and fixtures."""
import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--db-password",
        action="store",
        default=None,
        help="Database password for database connection tests"
    )


@pytest.fixture
def db_password(request):
    """Fixture to get database password from command line."""
    return request.config.getoption("--db-password")

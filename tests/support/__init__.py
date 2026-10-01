"""Stand-ins for the values that live in .env and nowhere else.

The real PIN, phone numbers and owner's name are never written into a tracked
file. Tests use these instead, and conftest installs TEST_PIN and TEST_NAME as
the configured values for every test, so nothing here can pass or fail on what
the live ones happen to be. `test_no_live_values.py` fails if a live value
ever lands in the tree.
"""

TEST_NAME = "Sam"
TEST_PIN = "4826"
TEST_PIN_SPOKEN = "four eight two six"
TEST_OWNER = "+15555550100"    # 555-0100..0199 is reserved for fiction

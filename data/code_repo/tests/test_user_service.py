"""User service tests."""

class MockDb:
    def __init__(self):
        self.add_called = False
        self.commit_called = False

    def add(self, user):
        self.add_called = True

    def commit(self):
        self.commit_called = True

def test_mock_db_tracks_calls():
    db = MockDb()
    db.add({"name": "Test User"})
    db.commit()
    assert db.add_called and db.commit_called

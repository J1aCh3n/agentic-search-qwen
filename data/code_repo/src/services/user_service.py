"""User service."""

class UserService:
    def __init__(self, db):
        self.db = db

    def create_user(self, user):
        self.db.add(user)
        self.db.commit()
        return user

    def get_user(self, user_id):
        return self.db.query("User").filter(id=user_id).first()

"""Project service."""

class ProjectService:
    def get_project(self, project_id: str) -> dict:
        return {"project_id": project_id}

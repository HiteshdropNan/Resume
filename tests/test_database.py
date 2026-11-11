import os
import pytest
import tempfile
from datetime import datetime
from src.database.models import JobListing, ApplicationStatus
from src.database.tracker import JobTracker

def test_database_initialization_and_operations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        tracker = JobTracker(db_path=db_path)
        assert os.path.exists(db_path)

        # Test insert
        job = JobListing(
            source="greenhouse",
            url="https://boards.greenhouse.io/sample/jobs/123",
            title="Quantitative Researcher",
            company="Citadel Securities",
            location="Remote",
            fit_score=90,
            status=ApplicationStatus.MATCHED
        )
        tracker.add_or_update_job(job)

        # Test job_exists
        assert tracker.job_exists("https://boards.greenhouse.io/sample/jobs/123") is True
        assert tracker.job_exists("https://boards.greenhouse.io/sample/jobs/999") is False

        # Test get_pending_jobs
        pending = tracker.get_pending_jobs(min_score=80)
        assert len(pending) == 1
        assert pending[0].company == "Citadel Securities"
        assert pending[0].fit_score == 90

        # Test update_status
        tracker.update_status(job.url, ApplicationStatus.APPLIED)
        stats = tracker.get_stats()
        assert stats["applied"] == 1
        assert stats["total"] == 1
    finally:
        import gc
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

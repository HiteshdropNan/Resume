import os
import tempfile
import pytest
from src.discovery.remote_finder import RemoteJobFinder
from src.customizer.resume_tailor import ResumeTailorAgent
from src.database.tracker import JobTracker
from src.database.models import JobListing, ApplicationStatus

def test_remote_detection():
    finder = RemoteJobFinder()
    assert finder._is_remote("Remote - US") is True
    assert finder._is_remote("London, UK", "We offer a 100% remote working model") is True
    assert finder._is_remote("Anywhere / Global") is True
    assert finder._is_remote("Chicago, IL", "Onsite 5 days a week") is False

def test_resume_tailoring_and_archetypes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tailor = ResumeTailorAgent()
        
        # Test 1: Quant Developer JD (C++ / low-latency focus)
        dev_res = tailor.customize_for_job(
            title="Senior C++ Quantitative Developer",
            company="FalconX",
            description="Looking for modern C++20 expert with low-latency multithreading and Boost.Asio.",
            job_url="https://falconx.io/careers/1",
            output_dir=tmp_dir
        )
        assert dev_res["archetype"] == "QUANT_DEVELOPER"
        assert os.path.exists(dev_res["markdown"])
        assert os.path.exists(dev_res["html"])
        assert os.path.exists(dev_res["kit"])

        with open(dev_res["markdown"], "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "Modern C++" in md_content
            assert "WorldQuant Brain" in md_content

        # Test 2: Quant Trader JD (Order book / microstructure focus)
        trader_res = tailor.customize_for_job(
            title="Quantitative Trader",
            company="Wintermute",
            description="Market making and statistical arbitrage with order book microstructure and execution logic.",
            job_url="https://wintermute.com/careers/2",
            output_dir=tmp_dir
        )
        assert trader_res["archetype"] == "QUANT_TRADER_MFT"

def test_manual_tracker_workflow():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        tracker = JobTracker(db_path=db_path)
        job = JobListing(
            source="linkedin",
            url="https://linkedin.com/jobs/view/999",
            title="Quantitative Researcher",
            company="Susquehanna",
            location="Remote",
            is_remote=True,
            fit_score=92,
            status=ApplicationStatus.MATCHED
        )
        job_id = tracker.add_or_update_job(job)

        # Mark resume generated
        tracker.mark_resume_generated(job_id, "path/to/resume.html", "path/to/kit.md")
        fetched = tracker.get_job_by_id(job_id)
        assert fetched.status == ApplicationStatus.RESUME_GENERATED
        assert fetched.tailored_resume_path == "path/to/resume.html"

        # Mark applied manually
        tracker.mark_applied_manually(job_id)
        updated = tracker.get_job_by_id(job_id)
        assert updated.status == ApplicationStatus.APPLIED_MANUALLY
        assert updated.applied_at is not None
    finally:
        import gc
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

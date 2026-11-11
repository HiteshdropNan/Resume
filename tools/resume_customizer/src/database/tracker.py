import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, timezone
from src.database.models import JobListing, ApplicationStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class JobTracker:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(Path(db_path) if db_path else (PROJECT_ROOT / "data" / "applications.db"))
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    location TEXT,
                    description TEXT,
                    is_remote INTEGER DEFAULT 0,
                    fit_score INTEGER,
                    fit_reasoning TEXT,
                    tailored_resume_path TEXT,
                    application_kit_path TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    applied_at TIMESTAMP,
                    error_message TEXT
                )
            """)
            conn.commit()

            # Ensure columns exist in case of an older schema
            cursor.execute("PRAGMA table_info(jobs)")
            existing_cols = {col[1] for col in cursor.fetchall()}
            if "is_remote" not in existing_cols:
                cursor.execute("ALTER TABLE jobs ADD COLUMN is_remote INTEGER DEFAULT 0")
            if "tailored_resume_path" not in existing_cols:
                cursor.execute("ALTER TABLE jobs ADD COLUMN tailored_resume_path TEXT")
            if "application_kit_path" not in existing_cols:
                cursor.execute("ALTER TABLE jobs ADD COLUMN application_kit_path TEXT")
            conn.commit()
        finally:
            conn.close()

    def job_exists(self, url: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM jobs WHERE url = ?", (url,))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def add_or_update_job(self, job: JobListing) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (
                    source, url, title, company, location, description, 
                    is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                    application_kit_path, status, created_at, applied_at, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title,
                    company=excluded.company,
                    location=coalesce(excluded.location, jobs.location),
                    description=coalesce(excluded.description, jobs.description),
                    is_remote=coalesce(excluded.is_remote, jobs.is_remote),
                    fit_score=coalesce(excluded.fit_score, jobs.fit_score),
                    fit_reasoning=coalesce(excluded.fit_reasoning, jobs.fit_reasoning),
                    tailored_resume_path=coalesce(excluded.tailored_resume_path, jobs.tailored_resume_path),
                    application_kit_path=coalesce(excluded.application_kit_path, jobs.application_kit_path),
                    status=excluded.status,
                    applied_at=coalesce(excluded.applied_at, jobs.applied_at),
                    error_message=coalesce(excluded.error_message, jobs.error_message)
            """, (
                job.source,
                job.url,
                job.title,
                job.company,
                job.location,
                job.description,
                1 if job.is_remote else 0,
                job.fit_score,
                job.fit_reasoning,
                job.tailored_resume_path,
                job.application_kit_path,
                job.status.value if isinstance(job.status, ApplicationStatus) else job.status,
                job.created_at.isoformat() if isinstance(job.created_at, datetime) else job.created_at,
                job.applied_at.isoformat() if isinstance(job.applied_at, datetime) else job.applied_at,
                job.error_message
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def _row_to_job(self, r: tuple) -> JobListing:
        return JobListing(
            id=str(r[0]),
            source=r[1],
            url=r[2],
            title=r[3],
            company=r[4],
            location=r[5],
            description=r[6],
            is_remote=bool(r[7]),
            fit_score=r[8],
            fit_reasoning=r[9],
            tailored_resume_path=r[10],
            application_kit_path=r[11],
            status=ApplicationStatus(r[12]),
            created_at=datetime.fromisoformat(r[13]) if r[13] else datetime.now(timezone.utc),
            applied_at=datetime.fromisoformat(r[14]) if r[14] else None,
            error_message=r[15]
        )

    def get_job_by_id(self, job_id: Union[int, str]) -> Optional[JobListing]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, url, title, company, location, description, 
                       is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                       application_kit_path, status, created_at, applied_at, error_message
                FROM jobs WHERE id = ?
            """, (str(job_id),))
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
            return None
        finally:
            conn.close()

    def get_job_by_url(self, url: str) -> Optional[JobListing]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, url, title, company, location, description, 
                       is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                       application_kit_path, status, created_at, applied_at, error_message
                FROM jobs WHERE url = ?
            """, (url,))
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
            return None
        finally:
            conn.close()

    def get_remote_jobs(self, min_score: int = 70) -> List[JobListing]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, url, title, company, location, description, 
                       is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                       application_kit_path, status, created_at, applied_at, error_message
                FROM jobs
                WHERE is_remote = 1 AND (fit_score IS NULL OR fit_score >= ?)
                ORDER BY fit_score DESC, id DESC
            """, (min_score,))
            rows = cursor.fetchall()
            return [self._row_to_job(r) for r in rows]
        finally:
            conn.close()

    def get_all_jobs(self, limit: int = 50) -> List[JobListing]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, url, title, company, location, description, 
                       is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                       application_kit_path, status, created_at, applied_at, error_message
                FROM jobs
                ORDER BY fit_score DESC, id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [self._row_to_job(r) for r in rows]
        finally:
            conn.close()

    def mark_resume_generated(self, job_id_or_url: Union[int, str], resume_path: str, kit_path: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET status = ?, tailored_resume_path = ?, application_kit_path = ?
                WHERE id = ? OR url = ?
            """, (ApplicationStatus.RESUME_GENERATED.value, resume_path, kit_path, str(job_id_or_url), str(job_id_or_url)))
            conn.commit()
        finally:
            conn.close()

    def mark_applied_manually(self, job_id_or_url: Union[int, str]):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                UPDATE jobs 
                SET status = ?, applied_at = ?
                WHERE id = ? OR url = ?
            """, (ApplicationStatus.APPLIED_MANUALLY.value, now, str(job_id_or_url), str(job_id_or_url)))
            conn.commit()
        finally:
            conn.close()

    def update_status(self, url: str, status: ApplicationStatus, error_message: Optional[str] = None):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            applied_at = datetime.now(timezone.utc).isoformat() if status in (ApplicationStatus.APPLIED, ApplicationStatus.APPLIED_MANUALLY) else None
            cursor.execute("""
                UPDATE jobs 
                SET status = ?, applied_at = coalesce(?, applied_at), error_message = ?
                WHERE url = ?
            """, (status.value, applied_at, error_message, url))
            conn.commit()
        finally:
            conn.close()

    def get_pending_jobs(self, min_score: int = 70) -> List[JobListing]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source, url, title, company, location, description, 
                       is_remote, fit_score, fit_reasoning, tailored_resume_path, 
                       application_kit_path, status, created_at, applied_at, error_message
                FROM jobs
                WHERE status IN ('MATCHED', 'RESUME_GENERATED', 'PENDING_REVIEW') AND (fit_score IS NULL OR fit_score >= ?)
                ORDER BY fit_score DESC
            """, (min_score,))
            rows = cursor.fetchall()
            return [self._row_to_job(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT status, count(*) FROM jobs GROUP BY status")
            counts = dict(cursor.fetchall())
            cursor.execute("SELECT count(*) FROM jobs")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM jobs WHERE is_remote = 1")
            remote_count = cursor.fetchone()[0]
            return {
                "total": total,
                "remote_total": remote_count,
                "applied_manually": counts.get(ApplicationStatus.APPLIED_MANUALLY.value, 0),
                "resume_generated": counts.get(ApplicationStatus.RESUME_GENERATED.value, 0),
                "applied": counts.get(ApplicationStatus.APPLIED.value, 0),
                "matched": counts.get(ApplicationStatus.MATCHED.value, 0),
                "pending_review": counts.get(ApplicationStatus.PENDING_REVIEW.value, 0),
                "skipped": counts.get(ApplicationStatus.SKIPPED.value, 0),
                "failed": counts.get(ApplicationStatus.FAILED.value, 0),
            }
        finally:
            conn.close()

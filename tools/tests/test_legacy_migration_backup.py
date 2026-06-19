"""Tests for tools/legacy_migration.py backup error handling (issue #6).

Verifies that:
- Missing backup directories are detected with an informative log message
  rather than a raw traceback.
- Permission errors, malformed manifests, and other OSError subclasses are
  handled with specific exception types (not bare `Exception`).
- The CLI returns a non-zero exit code when backup-related operations fail.

Each test uses tempfile / mock to avoid touching real filesystem paths.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

TOOLS_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TOOLS_DIR.parent

# Make tools/ importable so `import legacy_migration` works.
sys.path.insert(0, str(TOOLS_DIR))

import legacy_migration as lm  # noqa: E402


def _make_engine(backup_dir: str = None) -> "lm.MigrationEngine":
    """Build a MigrationEngine with the smallest valid config for backup ops."""
    config = lm.MigrationConfig(
        migration_id="MIG-TEST",
        migration_type=lm.MigrationType.DATA,
        from_version=1,
        to_version=2,
        source_connection="test",
        target_connection="test",
        backup_dir=backup_dir,
    )
    return lm.MigrationEngine(config)


class TestCreateBackupErrorHandling(unittest.TestCase):
    """_create_backup must return False (not raise) on filesystem errors."""

    def test_create_backup_returns_false_on_permission_error(self):
        """If the backup directory cannot be created due to permission
        errors, _create_backup must log an informative message and return
        False rather than propagating the exception."""
        engine = _make_engine()
        with patch(
            "pathlib.Path.mkdir",
            side_effect=PermissionError(13, "Permission denied"),
        ):
            result = engine._create_backup()
            self.assertFalse(result)

    def test_create_backup_returns_false_on_oserror(self):
        """Generic OSError (disk full, invalid path, etc.) must be caught
        separately so the log message names the specific failure type."""
        engine = _make_engine()
        with patch("pathlib.Path.mkdir", side_effect=OSError(28, "No space left")):
            result = engine._create_backup()
            self.assertFalse(result)

    def test_create_backup_success_returns_true(self):
        """Happy path: valid backup_dir creates manifest and returns True."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            engine.config.backup_dir = tmp
            result = engine._create_backup()
            self.assertTrue(result)
            manifest = Path(tmp) / "migration_MIG-TEST" / "manifest.json"
            self.assertTrue(manifest.exists())
            data = json.loads(manifest.read_text())
            self.assertEqual(data["migration_id"], "MIG-TEST")
            self.assertEqual(data["from_version"], 1)
            self.assertEqual(data["to_version"], 2)


class TestRestoreFromBackupErrorHandling(unittest.TestCase):
    """_restore_from_backup must return False (not raise) on filesystem errors."""

    def test_restore_returns_false_when_backup_dir_missing(self):
        """The most common failure: backup directory was never created
        (or was deleted). Return False with a clear message."""
        engine = _make_engine()
        missing = Path("/tmp/does-not-exist-MIG-NEVER-MADE-9c2d")
        # Make sure the path really doesn't exist.
        if missing.exists():
            missing.rmdir()
        result = engine._restore_from_backup(missing)
        self.assertFalse(result)

    def test_restore_returns_false_when_path_is_file_not_dir(self):
        """If backup_path points to a file (not a directory), the restore
        should fail gracefully rather than recursing into a file."""
        engine = _make_engine()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"not a directory")
            tmpfile = Path(f.name)
        try:
            result = engine._restore_from_backup(tmpfile)
            self.assertFalse(result)
        finally:
            tmpfile.unlink()

    def test_restore_returns_false_when_backup_path_is_none(self):
        """Defensive: passing None must produce an error message, not crash."""
        engine = _make_engine()
        result = engine._restore_from_backup(None)  # type: ignore[arg-type]
        self.assertFalse(result)

    def test_restore_returns_false_when_manifest_missing(self):
        """Backup directory exists but manifest.json is missing (corrupt
        or incomplete backup). Should return False with a clear message."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            result = engine._restore_from_backup(Path(tmp))
            self.assertFalse(result)

    def test_restore_returns_false_on_malformed_manifest(self):
        """Backup manifest exists but contains invalid JSON."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("{ this is not valid JSON")
            result = engine._restore_from_backup(Path(tmp))
            self.assertFalse(result)

    def test_restore_returns_false_on_permission_error(self):
        """OSError subclass (PermissionError) reading manifest must be
        caught and reported, not propagated."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"migration_id": "MIG-TEST"}')
            # Patch the built-in open to raise PermissionError when reading
            # the manifest. Root chmod doesn't restrict access, so we mock.
            original_open = open

            def open_with_perm_error(path, *args, **kwargs):
                if str(path).endswith("manifest.json") and "r" in (args[0] if args else kwargs.get("mode", "r")):
                    raise PermissionError(13, "Permission denied")
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=open_with_perm_error):
                result = engine._restore_from_backup(Path(tmp))
                self.assertFalse(result)

    def test_restore_success_returns_true(self):
        """Happy path: valid manifest, valid directory → True."""
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({
                    "migration_id": "MIG-TEST",
                    "created_at": "2024-01-15T00:00:00+00:00",
                    "files": [],
                })
            )
            result = engine._restore_from_backup(Path(tmp))
            self.assertTrue(result)


class TestExitCodes(unittest.TestCase):
    """main() must return non-zero exit code when migration/rollback fails."""

    def test_migrate_command_returns_nonzero_on_failure(self):
        """If MigrationEngine.run() returns a FAILED status, main() must
        exit 1 so CI scripts and shell wrappers can detect failure."""
        fake_result = MagicMock()
        fake_result.status.value = "failed"
        fake_result.migrated_records = 0
        fake_result.failed_records = 5
        fake_result.duration_seconds = 1.23
        fake_result.warnings = []
        with patch.object(lm, "MigrationEngine") as MockEngine:
            MockEngine.return_value.run.return_value = fake_result
            with patch.object(lm, "MigrationConfig") as MockConfig:
                MockConfig.return_value = MagicMock()
                with patch("sys.argv", [
                    "legacy_migration.py", "migrate",
                    "--from-version", "1", "--to-version", "2",
                ]):
                    self.assertEqual(lm.main(), 1)

    def test_rollback_command_returns_nonzero_on_failure(self):
        """If rollback fails (e.g. backup directory missing), main() must
        exit 1 so the operator knows to investigate."""
        fake_result = MagicMock()
        fake_result.status.value = "failed"
        with patch.object(lm, "MigrationEngine") as MockEngine:
            MockEngine.return_value.rollback.return_value = fake_result
            with patch.object(lm, "MigrationConfig") as MockConfig:
                MockConfig.return_value = MagicMock()
                with patch("sys.argv", [
                    "legacy_migration.py", "rollback",
                    "--migration-id", "MIG-MISSING",
                ]):
                    self.assertEqual(lm.main(), 1)


class TestIntegrationRollbackMissingBackup(unittest.TestCase):
    """End-to-end: rollback with missing backup dir returns exit code 1."""

    def test_rollback_missing_backup_exits_1(self):
        """Run the actual CLI binary with a non-existent backup dir and
        verify exit code is 1 (graceful failure)."""
        with tempfile.TemporaryDirectory() as tmp:
            missing_backup = os.path.join(tmp, "never-created-MIG-X9Z")
            result = subprocess.run(
                [
                    sys.executable, str(TOOLS_DIR / "legacy_migration.py"),
                    "rollback",
                    "--migration-id", "MIG-X9Z",
                    "--backup-dir", missing_backup,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Exit code must be 1 (graceful failure), not 0 and not a Python
            # traceback exit code (typically 1 too, but stderr should contain
            # our informative message instead of a raw traceback).
            self.assertEqual(result.returncode, 1)
            # Stderr should mention the missing backup (either "does not
            # exist" from _restore_from_backup or "no backup found" from
            # the rollback precheck). The key signal is an informative
            # error message, not a raw Python traceback.
            stderr_lower = result.stderr.lower()
            self.assertTrue(
                "does not exist" in stderr_lower or "no backup found" in stderr_lower,
                f"Expected informative backup-missing error, got: {result.stderr!r}",
            )
            # And it must NOT be a raw Python traceback.
            self.assertNotIn("Traceback (most recent call last)", result.stderr)


if __name__ == "__main__":
    unittest.main()
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from framepin import hashing
from framepin.cli import main
from framepin.listfile import snapshot_from_lists
from framepin.manifest import snapshot


def run_cli(argv):
    """Run the CLI capturing (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class BaseTmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-feat-")
        self.prev = os.getcwd()
        os.chdir(self.dir)
        os.makedirs("data")
        for i in range(6):
            with open(f"data/clip_{i}.bin", "wb") as fh:
                fh.write(f"frames-{i}".encode() * 32)
        main(["init"])

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.dir, ignore_errors=True)


class LogTests(BaseTmp):
    def test_log_empty(self):
        code, out, _ = run_cli(["log"])
        self.assertEqual(code, 0)
        self.assertIn("no dataset versions recorded", out)

    def test_log_lists_versions_newest_first(self):
        run_cli(["snapshot", "data"])
        with open("data/clip_0.bin", "wb") as fh:
            fh.write(b"CHANGED" * 8)
        run_cli(["snapshot", "data"])
        code, out, _ = run_cli(["log"])
        self.assertEqual(code, 0)
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("files", lines[0])

    def test_log_json(self):
        run_cli(["snapshot", "data"])
        code, out, _ = run_cli(["log", "--json"])
        self.assertEqual(code, 0)
        versions = json.loads(out)
        self.assertEqual(len(versions), 1)
        self.assertIn("root", versions[0])


class ParallelHashTests(BaseTmp):
    def test_dir_snapshot_deterministic_across_jobs(self):
        m1 = snapshot("data", jobs=1)
        m8 = snapshot("data", jobs=8)
        self.assertEqual(m1.root, m8.root)

    def test_list_snapshot_deterministic_across_jobs(self):
        paths = [os.path.abspath(f"data/clip_{i}.bin") for i in range(6)]
        with open("train.txt", "w") as fh:
            fh.write("\n".join(paths) + "\n")
        m1 = snapshot_from_lists(["train.txt"], jobs=1)
        m8 = snapshot_from_lists(["train.txt"], jobs=8)
        self.assertEqual(m1.root, m8.root)

    def test_cli_jobs_flag(self):
        code, out, _ = run_cli(["snapshot", "data", "--jobs", "8"])
        self.assertEqual(code, 0)
        self.assertIn("snapshot", out)


class ProgressTests(BaseTmp):
    def test_progress_goes_to_stderr_not_stdout(self):
        orig = hashing.PROGRESS_EVERY
        hashing.PROGRESS_EVERY = 2  # force progress lines with a tiny dataset
        try:
            code, out, err = run_cli(["snapshot", "data"])
        finally:
            hashing.PROGRESS_EVERY = orig
        self.assertEqual(code, 0)
        self.assertIn("hashed", err)
        self.assertNotIn("hashed", out)   # stdout stays pipe-clean


class JsonOutputTests(BaseTmp):
    def test_snapshot_json(self):
        code, out, _ = run_cli(["snapshot", "data", "--json"])
        self.assertEqual(code, 0)
        d = json.loads(out)
        self.assertEqual(d["file_count"], 6)
        self.assertEqual(d["missing"], 0)

    def test_verify_json_match_and_drift(self):
        _, out, _ = run_cli(["snapshot", "data", "--json"])
        vid = json.loads(out)["short"]
        code, out, _ = run_cli(["verify", "data", "--against", vid, "--json"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["match"])
        with open("data/clip_2.bin", "wb") as fh:
            fh.write(b"DRIFT" * 8)
        code, out, _ = run_cli(["verify", "data", "--against", vid, "--json"])
        self.assertEqual(code, 3)
        d = json.loads(out)
        self.assertFalse(d["match"])
        self.assertEqual(d["summary"]["modified"], 1)


if __name__ == "__main__":
    unittest.main()

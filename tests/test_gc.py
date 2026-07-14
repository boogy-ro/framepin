import contextlib
import io
import os
import shutil
import tempfile
import time
import unittest

import framepin
from framepin.cli import main
from framepin.repo import Repo


def run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


class GcTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="framepin-gc-")
        self.prev = os.getcwd()
        os.chdir(self.dir)
        os.makedirs("data")
        main(["init"])
        self.repo = Repo.discover(".")

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.dir, ignore_errors=True)

    def make_versions(self, n):
        """n distinct snapshots (created_at must differ for deterministic order)."""
        for i in range(n):
            with open("data/clip.bin", "wb") as fh:
                fh.write(f"content-{i}".encode() * 8)
            main(["snapshot", "data"])
            time.sleep(0.01)

    def test_dry_run_keeps_files(self):
        self.make_versions(7)
        code, out = run_cli(["gc", "--keep", "2"])
        self.assertEqual(code, 0)
        self.assertIn("would prune", out)
        self.assertEqual(len(self.repo.list_manifests()), 7)  # nothing deleted

    def test_apply_prunes_beyond_keep(self):
        self.make_versions(7)
        code, out = run_cli(["gc", "--keep", "2", "--apply"])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.repo.list_manifests()), 2)

    def test_run_referenced_manifests_survive(self):
        # a run pins the FIRST version; later versions are unreferenced
        with open("data/clip.bin", "wb") as fh:
            fh.write(b"pinned-version" * 8)
        with framepin.track(name="keeper", repo=self.repo) as run:
            pinned_root = run.use_dataset("data")
        self.make_versions(4)
        code, _ = run_cli(["gc", "--keep", "0", "--apply"])
        self.assertEqual(code, 0)
        survivors = self.repo.list_manifests()
        self.assertIn(pinned_root, survivors)      # referenced -> never pruned
        self.assertEqual(len(survivors), 1)

    def test_nothing_to_prune(self):
        self.make_versions(2)
        code, out = run_cli(["gc", "--keep", "5"])
        self.assertEqual(code, 0)
        self.assertIn("nothing to prune", out)


if __name__ == "__main__":
    unittest.main()

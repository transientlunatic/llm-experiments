import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from igor import condor
from igor.cli import main

PAPER = ("Title: A toy paper\n\nWe analyse the merger with nested sampling and 2000 live points.\n\n"
         "The total mass of the binary is 65 solar masses, from components of 30 and 35.\n\n"
         + "Further discussion of the noise model and the priors used in the analysis follows. " * 60)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.old, self.tmp = os.getcwd(), tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("paper.txt").write_text(PAPER)
        Path("prompts.jsonl").write_text('{"prompt": "hi"}\n')

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()

    def invoke(self, *args, ok=True):
        r = self.runner.invoke(main, list(args))
        if ok:
            self.assertEqual(r.exit_code, 0, r.output)
        return r

    # ------------------------------------------------------------- review
    def test_review_build_matrix_and_dag(self):
        r = self.invoke("review", "build", "paper.txt")
        self.assertIn("9 reviews = 3 models x 3 personas in 3 job(s)", r.output)
        run = Path("runs/paper")
        dag = (run / "review.dag").read_text()
        self.assertIn("JOB C claims.sub", dag)
        self.assertIn("PARENT C CHILD R1 R2 R3", dag)
        self.assertIn("FINAL E editor.sub", dag)
        self.assertIn("SCRIPT POST C /bin/sh", dag)
        plan = json.loads((run / "plan.json").read_text())
        pairs = {(x["model"], x["persona"]) for x in plan["referees"]}
        self.assertEqual(len(pairs), 9)                                 # every persona on every model
        self.assertEqual(len({x["model"] for x in plan["referees"]}), 3)
        sub = (run / "referee_R1.sub").read_text()
        self.assertIn("executable              = /bin/bash", sub)
        self.assertIn("transfer_executable     = False", sub)
        self.assertIn("job.sh", sub)
        self.assertIn("claim_findings.md", sub)
        self.assertIn("reviews/review_1.md", sub)
        self.assertIn("ref_rating.txt", sub)
        self.assertEqual(json.loads((run / "spec_R1.json").read_text())["ref_prompt"], "ref_rating.txt")
        self.assertIn("require_gpus", sub)
        # worker scripts referenced by absolute path actually exist
        for line in sub.splitlines():
            if line.startswith("transfer_input_files"):
                for f in line.split("=", 1)[1].split(","):
                    f = f.strip()
                    if f.startswith("/"):
                        self.assertTrue(Path(f).exists(), f)

    def test_review_options(self):
        self.invoke("review", "build", "paper.txt", "--name", "x", "--no-claims", "--split",
                    "--personas", "hostile,pedantic", "-n", "2", "--out-dir", "out", "--no-network")
        run = Path("out/x")
        dag = (run / "review.dag").read_text()
        self.assertNotIn("claims", dag)
        self.assertEqual(dag.count("JOB R"), 4)                          # split: model x persona nodes
        self.assertFalse((run / "claims.sub").exists())
        self.assertNotIn("claim_findings.md", (run / "referee_R1_1.sub").read_text())
        self.assertIn("antagonistic", (run / "focus_1.txt").read_text())      # first persona = hostile
        self.assertIn("meticulous pedant", (run / "focus_2.txt").read_text())          # second = pedantic

    def test_review_submit_and_run_dry_run(self):
        self.invoke("review", "build", "paper.txt")
        r = self.invoke("review", "submit", "runs/paper", "--dry-run")
        self.assertIn("condor_submit_dag -f review.dag", r.output)
        r = self.invoke("review", "run", "paper.txt", "--name", "y", "--dry-run")
        self.assertIn("condor_submit_dag -f review.dag", r.output)
        self.assertTrue(Path("runs/y/review.dag").exists())

    def test_submit_without_condor_is_a_clean_error(self):
        self.invoke("review", "build", "paper.txt")
        r = self.invoke("review", "submit", "runs/paper", ok=False)
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("HTCondor submit host", r.output)
        self.assertNotIn("Traceback", r.output)

    def test_unknown_persona_and_too_small_gpu(self):
        r = self.invoke("review", "build", "paper.txt", "--personas", "grumpy", ok=False)
        self.assertIn("unknown persona", r.output)
        r = self.invoke("review", "build", "paper.txt", "--vram", "1000", ok=False)
        self.assertIn("no catalogue model fits", r.output)

    def test_review_status(self):
        self.invoke("review", "build", "paper.txt")
        r = self.invoke("review", "status", "runs/paper")
        self.assertIn("reviews:  0/9", r.output)
        Path("runs/paper/reviews/review_1.md").write_text("x")
        Path("runs/paper/report.md").write_text("x")
        r = self.invoke("review", "status", "runs/paper")
        self.assertIn("1/9", r.output)
        self.assertIn("READY", r.output)

    def test_tarball_input(self):
        Path("src").mkdir()
        Path("src/main.tex").write_text(r"\documentclass{article}\begin{document}" + PAPER + r"\end{document}")
        with tarfile.open("2401.00001", "w:gz") as t:
            t.add("src/main.tex", arcname="main.tex")
        self.invoke("review", "build", "2401.00001", "--name", "tar", "--no-claims")
        self.assertIn("total mass", Path("runs/tar/doc.txt").read_text())

    # ------------------------------------------------------------- other jobs
    def test_batch_and_serve_build_only(self):
        r = self.invoke("batch", "llama3.1:8b", "prompts.jsonl", "--build-only")
        self.assertIn("GPU memory >=", r.output)
        sub = Path("runs/batch-prompts/batch.sub").read_text()
        self.assertIn("generate.py", sub)
        self.assertIn("prompts.jsonl", sub)
        self.assertTrue(Path("runs/batch-prompts/prompts.jsonl").exists())
        r = self.invoke("serve", "qwen2.5:14b", "--build-only", "--hours", "2")
        sub = Path("runs/serve-qwen2_5_14b/serve.sub").read_text()
        self.assertIn("serve", sub)
        self.assertIn("> 7200", sub)
        r = self.invoke("serve", "some/unknown:model", "--build-only")
        self.assertIn("not in the catalogue", r.output)

    def test_batch_dry_run_prints_submit(self):
        r = self.invoke("batch", "llama3.1:8b", "prompts.jsonl", "--dry-run")
        self.assertIn("condor_submit batch.sub", r.output)

    def test_connect_dry_run(self):
        r = self.invoke("connect", "123.0", "--dry-run", "--local-port", "9999")
        self.assertIn("condor_ssh_to_job", r.output)
        self.assertIn("9999", r.output)

    def test_models_listing(self):
        r = self.invoke("models", "--vram", "23000", "--doc", "paper.txt")
        self.assertIn("qwen2.5:14b", r.output)
        self.assertIn("does not fit", r.output)                          # 32b etc. exceed 23 GB budget at this ctx
        r = self.invoke("models", "--vram", "31000", "--tokens", "8000")
        self.assertIn("qwen2.5:32b", r.output)


class HelperTest(unittest.TestCase):
    def test_quote_args(self):
        self.assertEqual(condor.quote_args("a", "b c", "it's", 'say "x"'), '"a \'b c\' it\'\'s \'say ""x""\'"')
        self.assertEqual(condor.quote_args('q"q'), '"q""q"')

    def test_sub_has_required_pieces(self):
        s = condor.sub(model="m", command=["python3", "x.py"], gpu_mb=16000, name="n")
        self.assertIn("require_gpus            = (GlobalMemoryMb >= 16000)", s)
        self.assertTrue(s.rstrip().endswith("queue"))


if __name__ == "__main__":
    unittest.main()

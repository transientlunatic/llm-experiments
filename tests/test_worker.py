"""Worker-side logic, with the model stubbed out (worker scripts are flat modules, as on a node)."""
import sys
import tempfile
import unittest
from pathlib import Path

WORKER = Path(__file__).parent.parent / "igor" / "worker"
sys.path.insert(0, str(WORKER))
import review_pdf  # noqa: E402

DATA = WORKER.parent / "data" / "prompts"
BLOCK = ("REF_ASSESSMENT\noriginality: 3\nsignificance: 2\nrigour: 3\n"
         "overall_distribution: 4*=5, 3*=45, 2*=40, 1*=10, U=0")


class Stub:
    """Records every model call; behaviour is configured per test."""
    def __init__(self, truncate_review=False, rating_replies=None):
        self.calls, self.truncate_review = [], truncate_review
        self.rating_replies = list(rating_replies or [BLOCK])

    def chat_meta(self, content, images=None, *, num_ctx, num_predict=3000, temperature=0.2, label=""):
        self.calls.append({"label": label, "num_predict": num_predict, "prompt": content})
        truncated = self.truncate_review and label == "review"
        return {"text": f"[{label} text]", "truncated": truncated}

    def chat(self, content, images=None, *, num_ctx, num_predict=3000, temperature=0.2, label=""):
        self.calls.append({"label": label, "num_predict": num_predict, "prompt": content})
        return self.rating_replies.pop(0) if label == "rating" else f"[{label} text]"


class ReviewRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.prompt = self.dir / "p.txt"; self.prompt.write_text("Review it.")
        self.ref = DATA / "ref_rating.txt"
        self.focus = self.dir / "f.txt"; self.focus.write_text("Stance: grumpy.")
        self.out = self.dir / "r.md"
        self._orig = (review_pdf.chat, review_pdf.chat_meta)

    def tearDown(self):
        review_pdf.chat, review_pdf.chat_meta = self._orig
        self.tmp.cleanup()

    def run_review(self, stub, text, num_ctx=32768, max_out=2500):
        review_pdf.chat, review_pdf.chat_meta = stub.chat, stub.chat_meta
        review_pdf.run("doc.txt", str(self.prompt), str(self.focus), None, str(self.out), num_ctx, max_out,
                       text=text, ref_prompt=str(self.ref))
        return self.out.read_text()

    def test_rating_is_a_separate_call_and_is_appended(self):
        stub = Stub()
        out = self.run_review(stub, "word " * 400)
        self.assertEqual([c["label"] for c in stub.calls], ["review", "rating"])
        self.assertIn("REF_ASSESSMENT", out)
        self.assertNotIn("REF_ASSESSMENT", stub.calls[0]["prompt"])      # the review prompt no longer asks for it
        self.assertEqual(stub.calls[1]["num_predict"], 300)
        self.assertIn("Stance: grumpy.", stub.calls[1]["prompt"])        # rating sees the referee's stance

    def test_truncated_review_is_marked_but_keeps_its_rating(self):
        out = self.run_review(Stub(truncate_review=True), "word " * 400)
        self.assertIn("cut off at the output-length limit", out)
        self.assertIn("overall_distribution", out)

    def test_rating_retries_once_then_gives_up_cleanly(self):
        stub = Stub(rating_replies=["I think it is good.", BLOCK])
        self.assertIn("overall_distribution", self.run_review(stub, "word " * 400))
        self.assertEqual(sum(c["label"] == "rating" for c in stub.calls), 2)
        stub = Stub(rating_replies=["nope", "still nope"])
        out = self.run_review(stub, "word " * 400)
        self.assertNotIn("REF_ASSESSMENT", out)                          # no block, but the review is intact
        self.assertIn("[review text]", out)

    def test_chunked_path_uses_brief_partials_and_one_rating(self):
        stub = Stub()
        text = "\n\n".join(("Paragraph %d. " % i) + "word " * 300 for i in range(40))   # ~ 14k tokens
        out = self.run_review(stub, text, num_ctx=8192, max_out=2500)
        labels = [c["label"] for c in stub.calls]
        chunks = [c for c in stub.calls if c["label"].startswith("chunk")]
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c["num_predict"], review_pdf.PARTIAL_OUT)   # short outputs for partials
            self.assertIn("BRIEF bullet points", c["prompt"])
            self.assertIn("do not give a star rating", c["prompt"])
        self.assertEqual(labels.count("rating"), 1)
        self.assertIn("merge", labels)
        self.assertIn("overall_distribution", out)

    def test_prompt_files_are_consistent(self):
        review = (DATA / "review.txt").read_text()
        self.assertIn("at most about 800 words", review)
        self.assertNotIn("REF_ASSESSMENT", review)
        self.assertIn("REF_ASSESSMENT", (DATA / "ref_rating.txt").read_text())


if __name__ == "__main__":
    unittest.main()

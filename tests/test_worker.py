"""Worker-side logic, with the model stubbed out (worker scripts are flat modules, as on a node)."""
import sys
import tempfile
import unittest
from pathlib import Path

WORKER = Path(__file__).parent.parent / "igor" / "worker"
sys.path.insert(0, str(WORKER))
import claims  # noqa: E402
import doc  # noqa: E402
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


class AppendixTest(ReviewRunTest):
    MAIN = "\n\n".join(f"Main paragraph {i}. " + "science " * 200 for i in range(6))
    APP = "Appendix proof details. " + "lemma " * 1000

    def doc_text(self):
        return self.MAIN + "\n\n" + doc.APPENDIX_MARK + "\n" + self.APP

    def test_split_roundtrip(self):
        m, a = doc.split_main_appendix(self.doc_text())
        self.assertEqual(m, self.MAIN.rstrip())
        self.assertTrue(a.startswith("Appendix proof details"))
        self.assertEqual(doc.split_main_appendix("no appendix here"), ("no appendix here", ""))

    def test_latex_appendix_is_marked(self):
        tex = (r"\documentclass{article}\begin{document}Main body text. " + "x " * 200 +
               r"\appendix \section{Extra} Appendix content here.\end{document}")
        out = doc.clean_latex(tex)
        main, app = doc.split_main_appendix(out)
        self.assertIn("Main body text", main)
        self.assertNotIn("Appendix content", main)
        self.assertIn("Appendix content", app)

    def test_plain_text_heading_heuristic(self):
        body = "intro text. " * 400
        marked = doc.mark_appendix_plain(body + "\nAppendix A: Derivations\n" + "derivation " * 200)
        big = doc.mark_appendix_plain(body + "\nAppendices\n" + "derivation " * 5000)           # appendix > half the paper
        self.assertIn("derivation", doc.split_main_appendix(big)[1])
        self.assertNotIn("derivation", doc.split_main_appendix(big)[0])
        main, app = doc.split_main_appendix(marked)
        self.assertNotIn("derivation", main)
        self.assertIn("derivation", app)
        # a mention in the first half, or in running prose, must not trigger
        early = doc.mark_appendix_plain("Appendix A: not really\n" + "text " * 2000)
        self.assertEqual(doc.split_main_appendix(early)[1], "")
        prose = doc.mark_appendix_plain("text " * 1000 + "\nsee Appendix A for the full derivation of the result in detail now\n" + "tail " * 300)
        self.assertEqual(doc.split_main_appendix(prose)[1], "")

    def test_appendix_drop(self):
        text, _ = doc.load_document_ex(str(self.dir / "t.txt"), appendix="drop") if (self.dir / "t.txt").write_text(self.doc_text()) else (None, None)
        self.assertNotIn("lemma", text)
        text, _ = doc.load_document_ex(str(self.dir / "t.txt"))
        self.assertIn("lemma", text)

    def test_single_pass_labels_appendix_and_repeats_the_task_after_the_document(self):
        stub = Stub()
        self.run_review(stub, self.doc_text(), num_ctx=32768)
        prompt = stub.calls[0]["prompt"]
        self.assertTrue(prompt.index("--- MAIN TEXT ---") < prompt.index(review_pdf.APP_HEAD) < prompt.index("--- END OF DOCUMENT ---"))
        self.assertTrue(prompt.rstrip().endswith(review_pdf.REMINDER))                  # task restated last
        self.assertIn("Do not review the reference list", prompt)

    def test_appendix_is_truncated_or_omitted_to_protect_the_main_text(self):
        stub = Stub()
        self.run_review(stub, self.doc_text(), num_ctx=6500)                 # main fits, appendix only partly
        prompt = stub.calls[0]["prompt"]
        self.assertIn("Main paragraph 5", prompt)
        self.assertIn("[appendix truncated]", prompt)
        stub = Stub()
        self.run_review(stub, self.doc_text(), num_ctx=5900)                 # barely room for main: no appendix at all
        self.assertNotIn("lemma", stub.calls[0]["prompt"])

    def test_chunked_reading_never_includes_the_appendix(self):
        stub = Stub()
        self.run_review(stub, self.doc_text(), num_ctx=4096 + 1500)          # main text too long for one pass
        chunks = [c for c in stub.calls if c["label"].startswith("chunk")]
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertNotIn("lemma", c["prompt"])
            self.assertIn("appendices are omitted", c["prompt"])
            self.assertTrue(c["prompt"].rstrip().endswith(review_pdf.REMINDER))


class ClaimSelectionTest(unittest.TestCase):
    def test_citation_claims_are_capped(self):
        cl = [{"type": "citation", "quote": f"cite {i}", "importance": 5} for i in range(10)]
        cl += [{"type": "numerical", "quote": f"num {i}", "importance": 3} for i in range(10)]
        sel = claims.select(cl, 12)
        self.assertEqual(len(sel), 12)
        self.assertEqual(sum(c["type"] == "citation" for c in sel), 3)       # max(2, 12 // 4)
        self.assertEqual(sel[0]["type"], "citation")                         # still ranked by importance


if __name__ == "__main__":
    unittest.main()

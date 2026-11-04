"""Build the optional demo assets. Requires python-pptx (includes Pillow)."""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo_assets"
INK = "252B31"
MUTED = "58616C"
TEAL = "087F6D"
CORAL = "BE4D38"
WHITE = "FFFFFF"
PALE = "EAF4F1"
LINE = "DBE2E5"
FONT = "Trebuchet MS"


def color(value):
    return RGBColor.from_string(value)


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def diagram_box(draw, bounds, title, detail, accent=TEAL):
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=12, fill=f"#{PALE}", outline=f"#{LINE}", width=2)
    draw.rectangle((left, top + 12, left + 6, bottom - 12), fill=f"#{accent}")
    draw.text((left + 22, top + 22), title, font=font(29, True), fill=f"#{INK}")
    for index, line in enumerate(detail.split("\n")):
        draw.text((left + 22, top + 72 + index * 36), line, font=font(25), fill=f"#{MUTED}")


def arrow(draw, start, end):
    draw.line((start, end), fill=f"#{TEAL}", width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [end]
    for offset in (-0.5, 0.5):
        points.append((end[0] - 18 * math.cos(angle + offset), end[1] - 18 * math.sin(angle + offset)))
    draw.polygon(points, fill=f"#{TEAL}")


def build_diagrams():
    ASSETS.mkdir(exist_ok=True)
    image = Image.new("RGB", (1600, 670), f"#{WHITE}")
    draw = ImageDraw.Draw(image)
    draw.text((45, 28), "RAG: answering with an open book", font=font(43, True), fill=f"#{INK}")
    diagram_box(draw, (45, 125, 485, 330), "1. Find the notes", "Search our ten QA topics.\nKeep relevant sections.")
    diagram_box(draw, (580, 125, 1020, 330), "2. Ask the model", "Send question + notes\nto Groq.")
    diagram_box(draw, (1115, 125, 1555, 330), "3. Get an answer", "Use only those notes.\nAdmit missing information.")
    arrow(draw, (495, 225), (570, 225))
    arrow(draw, (1030, 225), (1105, 225))
    draw.text((45, 385), 'Question: "What is regression testing?"', font=font(34, True), fill=f"#{INK}")
    draw.text((45, 448), "Expected: check existing functionality after changes.", font=font(31), fill=f"#{MUTED}")
    draw.line((45, 530, 1555, 530), fill=f"#{LINE}", width=2)
    draw.text((45, 565), "LangChain connects the steps. LangSmith records what happened.", font=font(30), fill=f"#{TEAL}")
    image.save(ASSETS / "answer_flow.png")

    image = Image.new("RGB", (1600, 1040), f"#{WHITE}")
    draw = ImageDraw.Draw(image)
    draw.text((45, 25), "The active file connections", font=font(43, True), fill=f"#{INK}")
    diagram_box(draw, (45, 115, 735, 285), "day9_rag_golden_dataset.py", "Upload prepared RAG questions once.")
    diagram_box(draw, (865, 115, 1555, 285), "LangSmith dataset: rag-golden-v1", "Saved questions + reviewed expectations.")
    arrow(draw, (745, 200), (855, 200))
    diagram_box(draw, (865, 380, 1555, 550), "day10_eval_runner.py", "Read questions, run app, check results.")
    arrow(draw, (1210, 295), (1210, 370))
    diagram_box(draw, (45, 380, 735, 550), "day8_rag_app.py", "Use retrieved notes to request an answer.")
    arrow(draw, (855, 465), (745, 465))
    diagram_box(draw, (45, 645, 735, 815), "day8_rag_index.py", "Load, prepare, and search our QA notes.")
    arrow(draw, (390, 560), (390, 635))
    diagram_box(draw, (865, 645, 1555, 815), "Checks + LangSmith report", "Score retrieval and answers; pass or fail.")
    arrow(draw, (1210, 560), (1210, 635))
    draw.text((45, 865), "Notes: documents/qa_knowledge.txt", font=font(30, True), fill=f"#{INK}")
    draw.text((45, 925), "Groq supplies answer + AI-judge responses. Arrows show calls or data flow.", font=font(29), fill=f"#{MUTED}")
    image.save(ASSETS / "file_map.png")


def text(slide, left, top, width, height, value, size=23, bold=False, fill=INK):
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    for index, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(fill)
        paragraph.space_after = Pt(11)
    return shape


def rectangle(slide, left, top, width, height, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background()
    return shape


def new_slide(deck, title, section="THE MAIN STORY", subtitle=None):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(WHITE)
    rectangle(slide, 0.48, 0.38, 0.12, 0.28, TEAL)
    text(slide, 0.76, 0.36, 11.7, 0.4, section, size=12, bold=True, fill=TEAL)
    text(slide, 0.76, 0.97, 11.8, 0.72, title, size=33, bold=True)
    if subtitle:
        text(slide, 0.76, 1.8, 11.8, 0.65, subtitle, size=21, fill=MUTED)
    rectangle(slide, 0.76, 7.01, 11.8, 0.015, LINE)
    text(slide, 0.76, 7.12, 10.4, 0.24, "QA learning | Plain-language walkthrough | Recorded status: 4 Nov 2026", size=10, fill=MUTED)
    text(slide, 12.0, 7.09, 0.5, 0.3, str(len(deck.slides)), size=12, fill=MUTED)
    return slide


def table(slide, headers, rows, widths, top=2.15, height=4.5, size=17):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(0.76), Inches(top), Inches(11.8), Inches(height))
    content = shape.table
    for index, width in enumerate(widths):
        content.columns[index].width = Inches(width)
    for row_index, values in enumerate([headers] + rows):
        for column_index, value in enumerate(values):
            cell = content.cell(row_index, column_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(TEAL if row_index == 0 else (PALE if row_index % 2 else WHITE))
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = Inches(0.10)
            cell.margin_bottom = Inches(0.06)
            cell.text = value
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = FONT
                paragraph.font.size = Pt(size)
                paragraph.font.bold = row_index == 0
                paragraph.font.color.rgb = color(WHITE if row_index == 0 else INK)
    return shape


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def build_deck():
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    deck.core_properties.title = "Our First AI Quality Demo"
    deck.core_properties.subject = "A beginner-friendly guide to the QA assistant and its evaluations"
    deck.core_properties.author = "QA Learning"

    slide = new_slide(deck, "Can we trust the assistant's answer?", subtitle="Our first LangSmith evaluation framework")
    text(slide, 0.85, 2.85, 11.4, 1.35, "We built an assistant that uses our QA notes.\nThen we built a repeatable way to check its work.", size=29)
    rectangle(slide, 0.85, 4.65, 11.6, 1.2, PALE)
    text(slide, 1.1, 4.91, 11.0, 0.8, "Find the right information. Answer from it. Admit what is missing.", size=25, bold=True, fill=TEAL)
    notes(slide, "Ask: Have you ever seen an AI answer sound confident but be wrong? Explain that this is a learning project, not a finished production service. Main story: slides 1-11. Use appendices only for questions.")

    slide = new_slide(deck, "Six names, six different jobs")
    entries = [
        ("LLM", "The writer: generates text and can make mistakes."),
        ("Groq", "The service hosting the model we call."),
        ("LangChain", "The toolkit connecting model calls and documents."),
        ("LangSmith", "The activity record, test questions, and report book."),
        ("RAG", "Open-book answering: find notes before writing."),
        ("Evals", "Check and score the retrieved notes and answers."),
    ]
    for index, (label, detail) in enumerate(entries):
        left = 0.85 + (index % 2) * 6.0
        top = 2.0 + (index // 2) * 1.48
        text(slide, left, top, 5.45, 0.42, label, size=24, bold=True, fill=TEAL)
        text(slide, left, top + 0.49, 5.35, 0.85, detail, size=21)
    notes(slide, "LLM means large language model. RAG means retrieval-augmented generation. Groq, LangChain, and LangSmith are not three names for the same AI. Our current default model is openai/gpt-oss-20b hosted by Groq, not the OpenAI API.")

    slide = new_slide(deck, "Job A: answer with an open book")
    slide.shapes.add_picture(str(ASSETS / "answer_flow.png"), Inches(0.76), Inches(2.0), width=Inches(11.8))
    notes(slide, "Our earlier chat exercise asks the model directly. RAG first looks up the project's reference notes. These notes are supplied at request time; this is not training the model. Instructions reduce risk but do not guarantee correctness.")

    slide = new_slide(deck, "What should a good answer look like?")
    table(slide, ["Question", "What our notes contain", "Expected behavior"], [
        ["What is regression testing?", "Checks existing functionality after changes.", "Explain that idea in clear language."],
        ["What is a senior QA engineer's salary?", "No salary information.", "Say: I don't know based on the available knowledge."],
    ], [3.4, 4.0, 4.4], height=3.1, size=23)
    text(slide, 0.85, 5.72, 11.4, 0.8, "Refusing an unsupported question can be the correct result.", size=25, bold=True, fill=TEAL)
    notes(slide, "These are expected examples, not promises of a fresh model response. Wording can vary. A refusal should be judged against whether our knowledge base actually contains the answer.")

    slide = new_slide(deck, "Job B: check the assistant's work")
    for index, (title, detail) in enumerate([
        ("1. Prepared questions", "56 RAG examples with reviewed expectations."),
        ("2. Real answers", "Run the RAG assistant for each question."),
        ("3. Check and report", "Python checks + an AI judge + LangSmith results."),
    ]):
        top = 2.0 + index * 1.43
        text(slide, 0.85, top, 4.5, 0.5, title, size=25, bold=True, fill=TEAL)
        text(slide, 5.3, top, 6.9, 0.9, detail, size=23)
    notes(slide, "A golden dataset is the prepared question set. A trace records one request's steps. An experiment records one evaluated batch. The AI judge is another model request through Groq, not a person and not a guarantee of truth. This repeatable process is what we mean by a framework.")

    slide = new_slide(deck, "The scores are six questions about quality")
    table(slide, ["Score", "Plain question", "Target"], [
        ["Context recall", "Did we find the expected information?", ">= 0.85"],
        ["Context precision", "Are the retrieved notes relevant?", ">= 0.80"],
        ["Faithfulness", "Is the answer supported by the notes?", ">= 0.90"],
        ["Hallucination rate", "How much is unsupported? Lower is better.", "< 0.05"],
        ["No-answer rate", "Did it refuse unsupported questions?", ">= 0.95"],
        ["Answer relevance", "Did it address the question?", ">= 0.80"],
    ], [3.0, 7.0, 1.8], top=1.98, height=4.18, size=18)
    text(slide, 0.85, 6.35, 11.5, 0.5, "0.80 is a score, not necessarily 80% correct answers. Every target must pass.", size=18, fill=MUTED)
    notes(slide, "Scores average over applicable valid samples. Python checks expected context phrases and refusal wording; the AI judge estimates the other four scores. Day 10 does not check every stored reference field or show the reference answer to the judge. Any recorded generation/judge error also fails the gate. Human review remains important.")

    slide = new_slide(deck, "Offline tests are not a live AI quality pass")
    table(slide, ["Offline tests", "Live evaluation"], [
        ["Controlled or pretend model replies.", "Actual Groq-generated answers and AI judgments."],
        ["Checks our code's handling of success and failure.", "Checks answer quality on the dataset."],
        ["No Groq requests or quota needed.", "Needs Groq quota and LangSmith access."],
        ["19 unit tests pass.", "Latest complete pass is still unverified."],
    ], [5.9, 5.9], height=3.65, size=22)
    text(slide, 0.85, 6.15, 11.5, 0.6, "A command run on your laptop can still call online services.", size=24, bold=True, fill=TEAL)
    notes(slide, "There is also an offline retrieval check using actual notes and the cached local embedding model. Its phrase recall is 99.29% on 47 applicable examples, but it does not generate or judge real answers. Local versus GitHub is not the same distinction as offline versus live.")

    slide = new_slide(deck, "How the active files connect")
    slide.shapes.add_picture(str(ASSETS / "file_map.png"), Inches(2.64), Inches(1.81), height=Inches(4.97))
    notes(slide, "Follow the arrows from dataset upload to saved LangSmith questions, to the runner. The runner calls the RAG app, which uses the index. The index reads documents/qa_knowledge.txt. Results return to the runner's checks and the LangSmith experiment. The dataset uploader need not run for every evaluation. Other learning-day files are not all dependencies of Day 10.")

    slide = new_slide(deck, "What do we need to run it?")
    table(slide, ["Need", "Why"], [
        ["Python 3.12 + project packages", "Run our code and local embedding search."],
        ["QA notes + saved RAG dataset", "Provide reference information and prepared questions."],
        ["Groq account + private key + quota", "Generate answers and AI judge scores."],
        ["LangSmith account + private key", "Store questions, traces, and experiment reports."],
        ["Internet; GitHub secrets for CI", "Reach services; let automated runs authenticate safely."],
    ], [5.4, 6.4], height=4.1, size=20)
    notes(slide, "Use the Markdown guide or README for exact setup commands. Install requirements in a virtual environment. The embedding model downloads once and then runs locally. Configure private .env settings from .env.example. Never display keys. Upload approved RAG examples once if the cloud dataset is missing. GitHub CI uses the configured Groq organization quota, not an independent allowance.")

    slide = new_slide(deck, "A short demo that does not spend quota")
    text(slide, 0.85, 2.12, 11.5, 2.7, "1. Open the QA knowledge notes.\n2. Follow index -> RAG app -> evaluation runner.\n3. Run offline tests.\n4. Open a saved LangSmith question, sources, answer, and scores.", size=26)
    rectangle(slide, 0.85, 5.08, 11.6, 0.83, PALE)
    text(slide, 1.08, 5.27, 11.1, 0.45, "python -m unittest test_day10_eval_runner.py", size=24, bold=True, fill=TEAL)
    text(slide, 0.85, 6.15, 11.5, 0.5, "Label saved results as recorded evidence, not a new successful run.", size=20, fill=MUTED)
    notes(slide, "Spend about ten minutes on slides 1-11 and this short walkthrough. Do not run uploaders or the full live evaluator merely to fill demonstration time. Never open the private environment file when screen sharing.")

    slide = new_slide(deck, "What is complete, and what is pending?")
    text(slide, 0.85, 2.05, 5.4, 0.5, "Verified offline", size=27, bold=True, fill=TEAL)
    text(slide, 0.85, 2.79, 5.35, 2.5, "19 unit tests pass.\nRetrieval phrase recall: 99.29%.\nNo per-question recall regressions against the previous filter.", size=23)
    text(slide, 6.85, 2.05, 5.4, 0.5, "Still pending", size=27, bold=True, fill=CORAL)
    text(slide, 6.85, 2.79, 5.35, 2.5, "Full live quality pass.\nLatest precision >= 0.80.\nSuccessful live CI run with sufficient Groq quota.", size=23)
    text(slide, 0.85, 5.6, 11.5, 1.15, "The app answers. The framework checks.\nLangSmith shows the evidence. Humans review it.", size=25, bold=True, fill=TEAL)
    notes(slide, "The previous full run had 54 valid judge evaluations out of 56, hit daily quota, and scored 0.7954 precision before the latest retrieval changes. Quota failure is not proof of wrong answers, but incomplete results cannot be called a full pass. Do not lower thresholds. Any formal course completion decision depends on its acceptance criteria.")

    slide = new_slide(deck, "The active files: roles and dependencies", section="REFERENCE | USE ONLY WHEN ASKED")
    table(slide, ["File", "Role", "Without this role"], [
        ["day10_eval_runner.py", "Coordinates questions, app, scores, and gate.", "No end-to-end quality run."],
        ["day8_rag_app.py", "Requests an answer from retrieved notes.", "No active RAG answer flow."],
        ["day8_rag_index.py", "Loads, prepares, and searches notes.", "Cannot retrieve QA context."],
        ["day9_rag_golden_dataset.py", "Defines/uploads 56 RAG examples.", "No local test-set uploader; saved dataset can still run."],
        ["test_day10_eval_runner.py", "Tests the checker with controlled replies.", "Less protection against checker bugs."],
    ], [4.1, 4.25, 3.45], height=4.55, size=17)
    notes(slide, "These are the main files to understand first. Not every file needs to be executed separately. The runner imports the app; the app imports index helpers. Removing an imported file breaks the active path unless it is replaced.")

    slide = new_slide(deck, "Earlier exercises are not the active pipeline", section="REFERENCE | ALL OTHER APPLICATION PYTHON FILES")
    table(slide, ["File", "Why it exists", "Without it"], [
        ["day6_chat_app.py", "Interactive direct chat, without QA retrieval.", "No direct-chat exercise; RAG still works."],
        ["day7_evaluators.py", "Starter checks; own starter dataset.", "No Day 7 demo; Day 10 has its own checks."],
        ["day9_golden_dataset.py", "Defines/uploads 30 general-chat examples.", "No chat uploader; RAG uses a separate dataset."],
        ["day8_trace_check.py", "Checks LangSmith tracing without an LLM call.", "No dedicated connection smoke check."],
        ["day8_rag_app_backup.py", "Older RAG backup; not used by the runner.", "No effect on the active pipeline."],
    ], [4.1, 4.25, 3.45], height=4.55, size=17)
    notes(slide, "Day 7 uses qa-chat-golden-dataset, not chat-golden-v1. The Day 9 general-chat uploader uses chat-golden-v1; the RAG uploader uses rag-golden-v1. Avoid mixing them. A backup filename does not make that file the current application.")

    slide = new_slide(deck, "Supporting files: notes, setup, and automation", section="REFERENCE | NO NEED TO MEMORIZE")
    table(slide, ["File", "Purpose / what is missing without it"], [
        ["documents/qa_knowledge.txt", "Ten reference topics; no current knowledge source without it."],
        ["requirements.txt", "Application package list; harder to reproduce setup without it."],
        [".env.example / private .env", "Template / private settings; configure credentials another way if absent."],
        [".gitignore", "Excludes private/generated files; helps prevent accidental commits."],
        [".github/workflows/rag-evaluation.yml", "GitHub checks; local runs still work without this automation."],
        ["day8_manual_tests.md", "Historical manual test record; not application code."],
    ], [5.0, 6.8], height=4.55, size=17)
    notes(slide, "Never display .env or actual service credentials. CI means continuous integration: automated checks. The workflow runs offline checks on main pushes, pull requests to main, manual runs, and the schedule; live checks run on pull requests, manual runs, and the schedule, not main pushes. Required branch checks must be configured separately to block merges.")

    slide = new_slide(deck, "A small dictionary for the remaining words", section="REFERENCE | LOOK UP ONLY WHAT YOU NEED")
    table(slide, ["Word", "Plain meaning"], [
        ["Chunk / context", "One piece of notes / the notes supplied with a question."],
        ["Embedding / vector store", "Numbers for meaning comparison / searchable notes and numbers."],
        ["Prompt", "Instructions sent to the model."],
        ["Golden dataset", "Prepared questions with reviewed expectations."],
        ["Trace / experiment", "One request's recorded steps / one evaluated batch."],
        ["Token / quota", "A small unit of model text / an allowance for usage."],
        ["API / API key", "A way to contact a service / a private access credential."],
    ], [4.0, 7.8], height=4.55, size=18)
    notes(slide, "Tokens are not exactly words. The index is not a database of prewritten answers. Both generating answers and judging them consume model quota. Human review is needed because phrase checks and model judges have limitations.")

    slide = new_slide(deck, "Presentation and generated files", section="REFERENCE | OPTIONAL MATERIAL, NOT APP DEPENDENCIES")
    table(slide, ["File / folder", "Purpose / without it"], [
        ["README.md / DEMO_GUIDE.md", "Operations reference / this beginner explanation; code still runs without them."],
        ["DEMO_SLIDES.pptx", "Editable presentation; not required by the app or tests."],
        ["demo_assets/*.png", "Two visual diagrams; removing them affects the guide, not RAG."],
        ["tools/build_demo.py", "Regenerates slides and diagrams; optional presentation-only dependencies."],
        [".venv/ / __pycache__/ / .git/", "Installed environment / disposable cache / version history. Do not delete for a demo."],
    ], [4.65, 7.15], height=4.55, size=18)
    notes(slide, "The generator needs python-pptx and Pillow; they are not required for application evaluation and are not added to application requirements. Open the deck without running Python. All slides include speaker notes, while the two diagram images are also available beside the guide.")

    target = ROOT / "DEMO_SLIDES.pptx"
    deck.save(target)
    validate_deck(target)
    return target


def validate_deck(path):
    deck = Presentation(path)
    assert len(deck.slides) == 16
    assert sum(len(slide.notes_slide.notes_text_frame.text.strip()) > 0 for slide in deck.slides) == 16
    for slide_number, slide in enumerate(deck.slides, start=1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (slide_number, shape.name)
            assert shape.left + shape.width <= deck.slide_width, (slide_number, shape.name)
            assert shape.top + shape.height <= deck.slide_height, (slide_number, shape.name)
    print(f"Validated {len(deck.slides)} slides, speaker notes, and slide boundaries.")


if __name__ == "__main__":
    build_diagrams()
    output = build_deck()
    print(f"Created {output.name} and two PNG diagrams in {ASSETS.name}/")
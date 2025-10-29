import textwrap
from typing import Tuple, Any
import os
import zipfile
import shutil
import tempfile
from io import BytesIO
import mimetypes
import json
from typing import List, Optional
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
from google import genai
# Load env vars
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # LLMs
# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-1.5-flash")

SKIP_EXTENSIONS = ['.exe', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.mp4', '.mp3']
SKIP_MIME_PREFIXES = ['image', 'audio', 'video', 'application/zip', 'application/x-dosexec']

# ---------- Models ----------

class BugFix(BaseModel):
    bug: str
    fix: str
    file: Optional[str] = None

class FileSummary(BaseModel):
    filename: str
    summary: str

class AIAnalysisResponse(BaseModel):
    summary: str
    bugs: List[BugFix]
    files: List[FileSummary]
    suggestions: List[str]

# ---------- Helpers ----------

def is_valid_code_file(filepath):
    lpath = filepath.lower()
    if any(lpath.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    mime_type, _ = mimetypes.guess_type(filepath)
    return not mime_type or not any(mime_type.startswith(p) for p in SKIP_MIME_PREFIXES)

def extract_zip(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "project.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())
    extract_path = os.path.join(temp_dir, "project")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path
def extract_code_and_explanation(llm_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (code, explanation). Prefer the first triple-backtick block as code.
    """
    if not llm_output:
        return None, None

    # 1) Triple-backtick block
    m = re.search(r"```(?:\w+)?\n([\s\S]*?)```", llm_output)
    if m:
        code = m.group(1).rstrip()
        explanation = (llm_output[:m.start()] + llm_output[m.end():]).strip()
        explanation = re.sub(r'^(?:Here is the fixed code:|Fixed code:)\s*', '', explanation, flags=re.I).strip()
        return code, explanation if explanation else None

    # 2) Split by "Explanation" keyword
    split = re.split(r"\bExplanation\b:?", llm_output, flags=re.I)
    if len(split) >= 2:
        possible_code = split[0].strip()
        explanation = "Explanation: " + " ".join(s.strip() for s in split[1:])
        if re.search(r"[{};#<>]|def\s+|int\s+|#include|using\s+namespace|public\s+class", possible_code):
            return possible_code, explanation
        return None, llm_output.strip()

    # 3) Fallback: if the whole output looks like code, return it
    if len(llm_output) < 200_000 and re.search(r"[{};#<>]|def\s+|int\s+|#include", llm_output):
        return llm_output.strip(), None

    return None, llm_output.strip()


def zip_folder(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, arcname=rel_path)
    zip_buffer.seek(0)
    return zip_buffer

# ---------- LLM Core ----------

def _extract_json_object(text: str) -> Optional[str]:
    """Find the first {...} JSON-like substring in text and return it, else None."""
    m = re.search(r'(\{[\s\S]*\})', text)
    return m.group(1) if m else None

def is_schema_like(obj: Any) -> bool:
    """
    Heuristic to detect if the parsed JSON is a JSON-Schema / model schema
    rather than an instance. Checks for 'defs', '$schema', 'type' keys, etc.
    """
    if not isinstance(obj, dict):
        return False
    # common schema indicators
    if 'defs' in obj or '$schema' in obj:
        return True
    # top-level "type":"object" with properties is typical of JSON Schema
    if obj.get('type') == 'object' and obj.get('properties'):
        return True
    # pydantic model_json_schema often has 'title' + 'type' keys and 'definitions' or 'defs'
    if 'definitions' in obj:
        return True
    return False


def analyze_project_code(project_path: str, model_choice: str, max_retries: int = 1):
    # gather files
    all_code = {}
    for root, _, files in os.walk(project_path):
        for file in files:
            path = os.path.join(root, file)
            if is_valid_code_file(path) and os.path.getsize(path) < 50_000:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    rel_path = os.path.relpath(path, project_path)
                    all_code[rel_path] = content
                except:
                    continue

    # Build a simple example JSON instance (not a schema) to show the model what we want:
    example_instance = {
        "summary": "Short summary of the project (one or two sentences).",
        "bugs": [
            {
                "bug": "Brief description of bug found",
                "fix": "Short description of the fix",
                "file": "path/to/file.ext"
            }
        ],
        "files": [
            {
                "filename": "path/to/file.ext",
                "summary": "Short file-level summary"
            }
        ],
        "suggestions": [
            "List of improvement suggestions"
        ]
    }
    example_text = json.dumps(example_instance, indent=2)

    # Compose prompt (use dedent to avoid extra indentation)
    joined_code = "\n".join([f"# File: {name}\n{content}" for name, content in all_code.items()])
    prompt = textwrap.dedent(f"""
    Analyze the following project and produce a JSON object that exactly matches this structure.
    ONLY output the JSON object — no explanations, no schema, no extra text.

    Example JSON (use this structure exactly):
    ```json
    {example_text}
    ```

    Project files to analyze:
    {joined_code}

    Return a single JSON object following the example structure.
    """)

    # helper to call the chosen model backend
    def call_model_with_prompt(p: str) -> str:
        if model_choice == "Gemini":
            client = genai.Client()
            resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=p,
    )
            return getattr(resp, "text", getattr(resp, "output_text", str(resp)))
    # Try to get a valid instance, with simple retry if the model returns a schema-like object
    raw = call_model_with_prompt(prompt)

    # Try parse, with schema-detection fallback and retry
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # try to extract a {...} substring
        json_sub = _extract_json_object(raw)
        if json_sub:
            parsed = json.loads(json_sub)
        else:
            return f"Failed to parse JSON from model output. Raw output:\n{raw}"

    # If parsed looks like a schema (not an instance), retry once with short clarifying prompt
    if is_schema_like(parsed) and max_retries > 0:
        retry_prompt = textwrap.dedent("""
        You returned the JSON schema/structure. Please now OUTPUT ONLY a JSON *instance*
        that follows the example structure given earlier (use empty strings/lists where applicable).
        Do NOT output the schema again and do NOT add any explanation.
        """)
        raw2 = call_model_with_prompt(joined_code + "\n\n" + retry_prompt)
        try:
            parsed2 = json.loads(raw2)
        except json.JSONDecodeError:
            json_sub2 = _extract_json_object(raw2)
            if json_sub2:
                parsed2 = json.loads(json_sub2)
            else:
                return f"Retry failed to produce parseable JSON. Raw retry output:\n{raw2}"

        # If still schema-like, abort
        if is_schema_like(parsed2):
            return "Model repeatedly returned a schema instead of an instance. Raw retry output:\n" + raw2
        parsed = parsed2

    # At this point 'parsed' should be an instance; validate with pydantic
    try:
        return AIAnalysisResponse(**parsed)
    except Exception as e:
        # Show helpful debug info including raw model output
        return f"Pydantic validation failed: {e}\n\nRaw model output:\n{raw}"


def fix_code_file(filepath: str, model_choice: str, relpath: Optional[str]=None) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return None, f"read_failed: {e}"

    send_name = relpath or os.path.basename(filepath)
    # use file extension hint in fence (helps model produce correct syntax)
    ext = Path(filepath).suffix.lstrip('.') or "code"

    prompt = textwrap.dedent(f"""\
    You are a code fixer AI.
    Task: Fix bugs, clean up formatting, and improve code quality while preserving the language.
    IMPORTANT: Output ONLY the corrected source code inside a single triple-backtick code block.
    Do NOT write explanations outside the code block.
    Keep comments minimal and inline if necessary.
    Filename: {send_name}

    ```{ext}
    {code}
    ```
    """)

    try:
        if model_choice == "Gemini":
            client = genai.Client()
            response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
    )
            llm_output = getattr(response, "text", getattr(response, "output_text", str(response)))
    except Exception as e:
        return None, f"LLM call failed: {e}"

    code_only, explanation = extract_code_and_explanation(llm_output)

    # fallback heuristics
    if not code_only:
        if llm_output and re.search(r"[{};#<>]|def\s+|int\s+|#include|public\s+class", llm_output):
            code_only = llm_output.strip()
        else:
            # nothing safe to write back
            return None, llm_output.strip()

    return code_only, explanation

def fix_all_code(project_dir: str, model_choice: str) -> str:
    fixed_root = tempfile.mkdtemp()
    dest_root = os.path.join(fixed_root, "fixed_project")
    shutil.copytree(project_dir, dest_root, dirs_exist_ok=True)

    reports = []
    for root, _, files in os.walk(dest_root):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, dest_root)
            if not is_valid_code_file(filepath):
                continue
            try:
                if os.path.getsize(filepath) >= 50_000:
                    reports.append({"file": rel_path, "status": "skipped_large_file"})
                    continue
            except OSError:
                continue

            fixed_code, explanation = fix_code_file(filepath, model_choice, relpath=rel_path)
            if fixed_code:
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_code)
                    reports.append({"file": rel_path, "status": "fixed", "explanation": explanation or ""})
                except Exception as e:
                    reports.append({"file": rel_path, "status": f"write_failed: {e}"})
            else:
                reports.append({"file": rel_path, "status": "no_fix_obtained", "explanation": explanation or "No response or could not extract code."})

    # human-readable report
    report_txt_path = os.path.join(dest_root, "fix_report.txt")
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("Fix Report\n")
        f.write("="*40 + "\n\n")
        for r in reports:
            f.write(f"File: {r['file']}\nStatus: {r['status']}\n")
            if r.get("explanation"):
                f.write("Explanation / Notes:\n")
                f.write(r["explanation"].strip() + "\n")
            f.write("\n---\n\n")

    # JSON report
    report_json_path = os.path.join(dest_root, "fix_report.json")
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2)

    return dest_root

def generate_file_from_project(project_dir, filename, task_description, model_choice):
    prompt = f"You are an expert DevOps assistant. Based on this project structure, generate a `{filename}`:\n\n"
    for root, _, files in os.walk(project_dir):
        for file in files:
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, project_dir)
            if os.path.getsize(path) < 50_000:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    prompt += f"\n\n# {rel_path}\n```python\n{content[:3000]}\n```\n"
                except:
                    continue
    prompt += f"""
    ---
    Generate only the final `{filename}` content.
    {task_description}
    Only output the file content. No planning.
    """

    try:
        if model_choice == "Gemini":
            client = genai.Client()
            response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
    )

            return response.text
    except:
        return "❌ Error generating content."

# ---------- UI ----------

st.set_page_config(page_title="Project Fixer & Generator", layout="wide")
st.title("🛠️ Upload & Auto-Fix Project + Generate DevOps Files")

llm_option = "Gemini"  # Fixed to Gemini only


uploaded_file = st.file_uploader("📁 Upload your project as a .zip", type="zip")

if uploaded_file:
    with st.spinner("Extracting ZIP..."):
        project_path = extract_zip(uploaded_file)
        st.success("✅ Project extracted!")

    st.subheader("⚙️ Generate DevOps Files")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🐳 Generate Dockerfile & .dockerignore"):
            with st.spinner("Generating Docker files..."):
                dockerfile = generate_file_from_project(project_path, "Dockerfile", "Use Python as base image if applicable.", llm_option)
                dockerignore = generate_file_from_project(project_path, ".dockerignore", "Ignore unnecessary files.", llm_option)
                st.code(dockerfile, language='docker')
                st.code(dockerignore, language='bash')

    with col2:
        if st.button("📘 Generate README.md"):
            with st.spinner("Generating README.md..."):
                readme = generate_file_from_project(project_path, "README.md", "Include description, install, usage, license.", llm_option)
                st.code(readme, language='markdown')

    with col3:
        if st.button("🧾 Generate requirements.txt & .gitignore"):
            with st.spinner("Generating files..."):
                reqs = generate_file_from_project(project_path, "requirements.txt", "List dependencies.", llm_option)
                gitignore = generate_file_from_project(project_path, ".gitignore", "Common Python ignores.", llm_option)
                st.code(reqs, language='bash')
                st.code(gitignore, language='bash')

    st.subheader("🧹 Optional: Fix Codebase")
    if st.button("✨ Auto-Fix Entire Codebase"):
        with st.spinner(f"Fixing codebase using {llm_option}..."):
            fixed_path = fix_all_code(project_path, llm_option)
            zip_buf = zip_folder(fixed_path)
            st.success("✅ Code fixed!")
            st.download_button("📦 Download Fixed Project", data=zip_buf, file_name="fixed_project.zip", mime="application/zip")

    st.subheader("🧠 Analyze Codebase (Structured Output)")
    if st.button("🔍 Analyze with AI"):
        with st.spinner("Analyzing codebase..."):
            result = analyze_project_code(project_path, llm_option)
            if isinstance(result, str):
                st.error(f"❌ {result}")
            else:
                st.markdown(f"**📝 Summary:** {result.summary}")
                st.markdown("**🐛 Bugs & Fixes:**")
                for bug in result.bugs:
                    st.markdown(f"- `{bug.file}`: {bug.bug} ➜ _{bug.fix}_")
                st.markdown("**📂 File Summaries:**")
                for fs in result.files:
                    st.markdown(f"- `{fs.filename}`: {fs.summary}")
                st.markdown("**💡 Suggestions:**")
                for s in result.suggestions:
                    st.markdown(f"- {s}")

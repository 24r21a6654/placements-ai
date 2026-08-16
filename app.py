import io
import os
import traceback

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent

# --- 1. LLM ---
# NOTE: "gemma-4-31b-it" is not a valid Gemini Developer API model name.
# Use a real Gemini model (see https://ai.google.dev/gemini-api/docs/models for the current list).
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.3,
)

search_engine = DuckDuckGoSearchResults()


# --- 2. Tools ---
@tool
def job_search(role: str) -> str:
    """Search the web for current job openings matching a given role."""
    query = f"{role} job openings India 2026"
    return search_engine.invoke(query)


@tool
def skill_gap_analysis(role: str, resume_text: str) -> str:
    """Compare the student's resume skills against the requirements of a target role and list missing skills."""
    prompt = (
        f"You are a technical recruiter. Given this resume text:\n{resume_text}\n\n"
        f"And the target role: '{role}'\n\n"
        f"List the skills the candidate already has, and the skills they are missing "
        f"for this role. Be concise and use bullet points."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def project_ideas(missing_skills: str) -> str:
    """Suggest 3 practical project ideas to help a student build the given missing skills."""
    prompt = (
        f"Suggest 3 practical, resume-worthy project ideas that would help a student "
        f"learn and demonstrate these missing skills: {missing_skills}. "
        f"For each, give a one-line description."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def github_check(github_username: str) -> str:
    """Check a student's GitHub profile for recent public repo activity and languages used."""
    url = f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=5"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return f"Could not fetch GitHub data for user: {github_username}"
    repos = response.json()
    summary = [
        f"{repo['name']} (lang: {repo.get('language', 'N/A')}, updated: {repo['updated_at'][:10]})"
        for repo in repos
    ]
    return "Recent repos: " + "; ".join(summary) if summary else "No public repos found."


tools = [job_search, skill_gap_analysis, project_ideas, github_check]

career_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a Placement-Ready AI Career Agent for engineering students. "
        "Given a student's resume, target role, and GitHub username, use the available "
        "tools to: 1) find matching job openings, 2) identify skill gaps, "
        "3) suggest relevant projects, and 4) check their GitHub activity. "
        "Call multiple tools as needed before giving your final answer. "
        "End with a clear, structured summary."
    ),
)


def extract_final_text(agent_result: dict) -> str:
    for msg in reversed(agent_result.get("messages", [])):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                    return block["text"]
    return ""


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


def run_career_agent(resume_text: str, target_role: str, github_username: str) -> dict:
    query = (
        f"My target role is '{target_role}'. "
        f"My GitHub username is '{github_username}'. "
        f"Here is my resume:\n{resume_text}\n\n"
        f"Please find job openings, analyze my skill gaps, suggest projects, "
        f"and check my GitHub activity."
    )
    result = career_agent.invoke({"messages": [HumanMessage(content=query)]})
    tool_calls_made = [
        tc["name"]
        for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
    ]
    return {
        "student_role": target_role,
        "github_username": github_username,
        "tools_used": tool_calls_made,
        "final_summary": extract_final_text(result),
    }


# --- 3. FastAPI app ---
app = FastAPI(title="Placement-Ready AI Career Agent")

# Allow the frontend (hosted anywhere) to call this API.
# Tighten allow_origins to your actual frontend domain once you know it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/career-agent")
async def career_agent_endpoint(
    resume: UploadFile = File(..., description="Resume PDF file"),
    target_role: str = Form(...),
    github_username: str = Form(...),
):
    # Validate file type up front so we fail with a clean 400, not a crash.
    if resume.content_type not in ("application/pdf", "application/octet-stream") and not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file for the resume.")

    try:
        file_bytes = await resume.read()
        resume_text = extract_pdf_text(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF (it may be a scanned image).",
        )

    try:
        result = run_career_agent(resume_text, target_role, github_username)
        return JSONResponse(content=result)
    except Exception as exc:
        # Log the full traceback server-side (visible in Render logs) but
        # return a clean JSON error to the frontend instead of an HTML 500 page.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


"""
app.py — Jothidam Customer Support Assistant
ஜோதிட வாடிக்கையாளர் உதவி அமைப்பு

WORKFLOW:
1. Agent 1 answers using the local Jothidam book passages.
2. Agent 2 searches the web and prepares an independent answer.
3. Agent 3 records the question and both answers in a text file.
4. Streamlit displays both answers and the logging status.

Run the app using:
    streamlit run app.py
"""

# =========================================================
# BLOCK 1 — Load environment variables and import libraries
# =========================================================

from dotenv import load_dotenv

# Read API keys from .env without displaying them.
load_dotenv()

import streamlit as st
from pathlib import Path
from datetime import datetime

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Reuse the local PDF retrieval functions already built.
from search_knowledge import load_chunks, search_knowledge


# =========================================================
# BLOCK 2 — Configure the Streamlit page
# =========================================================

# Set the browser tab title, icon, and wide layout.
st.set_page_config(
    page_title="Jothidam Support | ஜோதிட உதவி",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# BLOCK 3 — Define application paths and shared resources
# =========================================================

# Keep output files in the same project folder as app.py.
PROJECT_FOLDER = Path(__file__).resolve().parent

# Agent 3 will append each question and both answers here.
LOG_FILE = PROJECT_FOLDER / "answers_log.txt"

# Configure the OpenAI model through CrewAI.
llm = LLM(model="openai/gpt-4o-mini")

# Create the web-search tool.
serper_tool = SerperDevTool()

# Load the prepared chunks once when the app starts.
@st.cache_resource
def get_knowledge_chunks():
    """Load and cache the local Jothidam text chunks."""
    return load_chunks()


# =========================================================
# BLOCK 4 — Create the three CrewAI agents
# =========================================================

# AGENT 1:
# Uses the retrieved passage from the Jothidam book.
# It must not pretend that an unsupported answer is in the book.

knowledge_agent = Agent(
    role="Jothidam Knowledge Assistant",
    goal=(
        "Answer the user's question using the supplied passage "
        "from the local Jothidam knowledge base."
    ),
  backstory=(
    "You are role-playing a professional traditional Jothidam "
    "astrologer with the equivalent of 30 years of study and "
    "consultation experience. You have deep familiarity with "
    "traditional Jothidam concepts, including Rasi, Nakshatra, "
    "Graha, Bhava, Dasa and Bhukti. "
    "Your role is to explain these concepts clearly, respectfully, "
    "and confidently to beginners and experienced learners. "
    "For this task, use the supplied passage from the user's "
    "Jothidam books as your primary and authoritative source. "
    "Base book-specific answers on the passage provided. "
    "If the passage does not contain enough information to answer "
    "the question, clearly state that the available book passage "
    "is insufficient rather than inventing information or "
    "claiming the book says something it does not say. "
    "Preserve Tamil terminology, traditional names, and meanings "
    "accurately. Answer in the user's language, Tamil or English, "
    "and provide a concise, professional, easy-to-understand answer. "
    "Do not claim to be a real human astrologer."
    ),
    llm=llm,
    verbose=False,
)

# AGENT 2:
# Always searches the web, independently of Agent 1's answer.

web_agent = Agent(
    role="Web Research Assistant",
    goal=(
        "Search the web for information relevant to the user's "
        "question and provide a separate, useful answer."
    ),
    backstory=(
        "You are a careful web researcher. Use the available "
        "web-search results as evidence. Do not invent facts, "
        "quotes, or URLs. Distinguish traditional beliefs from "
        "scientifically established facts when relevant. "
        "Mention useful source names or URLs when they are "
        "available in the search results. If the results are "
        "insufficient, say so. Answer in the user's language "
        "when possible."
    ),
    tools=[serper_tool],
    llm=llm,
    verbose=False,
)

# AGENT 3:
# Receives both earlier task outputs and prepares a record.
# Python will perform the actual file write after the Crew finishes.

entry_agent = Agent(
    role="Support Record Assistant",
    goal=(
        "Prepare a clear record containing the user's question "
        "and the answers from both previous agents."
    ),
    backstory=(
        "You maintain the support records for a bilingual "
        "Jothidam assistant. Preserve the original question "
        "and both answers. Do not change their meaning, "
        "invent information, or omit either answer."
    ),
    llm=llm,
    verbose=False,
)


# =========================================================
# BLOCK 5 — Retrieve relevant passages from local Jothidam notes
# =========================================================

def retrieve_local_passage(question):
    """
    Retrieve the best matching passages from the local Jothidam books.

    The updated search_knowledge() function returns a list of
    passages, so this function combines them for Agent 1.
    """

    chunks = get_knowledge_chunks()
    results = search_knowledge(question, chunks)

    # Handle an empty result list.
    if not results:
        return {
            "text": (
                "No matching passage was retrieved from the "
                "local Jothidam notes. Do not invent book content."
            ),
            "page": None,
            "chunk_id": None,
        }

    # Combine the retrieved passages and preserve their page numbers.
    passage_sections = []

    for number, result in enumerate(results, start=1):
        page = result.get("page")
        chunk_id = result.get("chunk_id", "Unknown")
        text = result.get("text", "")

        passage_sections.append(
            f"PASSAGE {number}\n"
            f"Page: {page if page is not None else 'Unknown'}\n"
            f"Chunk: {chunk_id}\n"
            f"Text:\n{text}"
        )

    combined_text = "\n\n".join(passage_sections)

    # Keep the first-ranked result's metadata for the UI and log.
    best_result = results[0]

    return {
        "text": combined_text,
        "page": best_result.get("page"),
        "chunk_id": best_result.get("chunk_id"),
    }


# =========================================================
# BLOCK 6 — Run all three agents sequentially
# =========================================================

def run_support_workflow(question):
    """
    Execute the three agents in the required order:

    Agent 1 -> local Jothidam answer
    Agent 2 -> web research answer
    Agent 3 -> record both answers

    All three tasks belong to one sequential Crew.
    """

    # Retrieve the local passage before asking Agent 1.
    passage = retrieve_local_passage(question)

    # -----------------------------------------------------
    # TASK 1 — Answer from the local Jothidam passage
    # -----------------------------------------------------

    local_task = Task(
        description=f"""
Answer the user's question using the supplied local passage.

USER QUESTION:
{question}

LOCAL JOTHIDAM PASSAGE:
{passage["text"]}

SOURCE PAGE:
{passage["page"] if passage["page"] is not None else "Unknown"}

INSTRUCTIONS:
- Use the passage as evidence.
- Do not use web research for this task.
- Do not invent facts or quotations.
- If the passage does not answer the question, say so.
- Give one clear, concise answer.
- Preserve Tamil words and names accurately.
- Answer in the user's language when possible.
""",
        expected_output=(
            "One concise answer based on the local passage, "
            "or a clear statement that the passage is insufficient."
        ),
        agent=knowledge_agent,
    )

    # -----------------------------------------------------
    # TASK 2 — Search the web and answer independently
    # -----------------------------------------------------

    web_task = Task(
        description=f"""
Research and answer this question using web search.

USER QUESTION:
{question}

INSTRUCTIONS:
- Use your web-search tool to find relevant information.
- This task must run after the local-answer task.
- Prepare an independent web-based answer.
- Do not simply copy the local answer.
- Do not invent facts, source names, or URLs.
- Distinguish web-based information from the book's content.
- If results are insufficient, state that clearly.
- Answer in the user's language when possible.
""",
        expected_output=(
            "One concise web-based answer with available source "
            "details, or a statement that evidence is insufficient."
        ),
        agent=web_agent,
    )

    # -----------------------------------------------------
    # TASK 3 — Prepare the support record
    # -----------------------------------------------------

    entry_task = Task(
        description=f"""
Prepare a plain-text support record for this interaction.

ORIGINAL USER QUESTION:
{question}

LOCAL BOOK SOURCE PAGE:
{passage["page"] if passage["page"] is not None else "Unknown"}

Include:
1. The date and time supplied below.
2. The original user question.
3. The complete answer from Agent 1.
4. The complete answer from Agent 2.

Record timestamp:
{datetime.now().astimezone().isoformat(timespec="seconds")}

Use the previous two task outputs as the authoritative answers.
Preserve their meaning and wording. Do not invent information.

Return the complete record as plain text.
""",
        expected_output=(
            "A complete plain-text record containing the question "
            "and both preceding agent answers."
        ),
        agent=entry_agent,

        # Explicitly pass the outputs from Task 1 and Task 2.
        context=[local_task, web_task],
    )

    # -----------------------------------------------------
    # Execute all three tasks in the specified order.
    # -----------------------------------------------------

    support_crew = Crew(
        agents=[
            knowledge_agent,
            web_agent,
            entry_agent,
        ],
        tasks=[
            local_task,
            web_task,
            entry_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    # kickoff() runs Task 1, then Task 2, then Task 3.
    support_crew.kickoff()

    # Extract the individual answers for the Streamlit UI.
    local_answer = (
        local_task.output.raw
        if local_task.output
        else "No local answer was returned."
    )

    web_answer = (
        web_task.output.raw
        if web_task.output
        else "No web answer was returned."
    )

    entry_text = (
        entry_task.output.raw
        if entry_task.output
        else "No support record was prepared."
    )

    # -----------------------------------------------------
    # Save the question and both answers to a text file.
    # -----------------------------------------------------

    # The actual file write is handled by Python, not left
    # to the language model. Each interaction is appended.
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 72 + "\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Question: {question}\n")
        log_file.write(f"Local source page: {passage['page']}\n")
        log_file.write("\n--- AGENT 1: LOCAL JOTHIDAM ANSWER ---\n")
        log_file.write(str(local_answer) + "\n")
        log_file.write("\n--- AGENT 2: WEB SEARCH ANSWER ---\n")
        log_file.write(str(web_answer) + "\n")
        log_file.write("\n--- AGENT 3: SUPPORT RECORD ---\n")
        log_file.write(str(entry_text) + "\n")

    # Return both answers and metadata to the interface.
    return {
        "question": question,
        "local_answer": str(local_answer),
        "web_answer": str(web_answer),
        "entry_text": str(entry_text),
        "source_page": passage["page"],
        "log_file": str(LOG_FILE),
    }


# =========================================================
# BLOCK 7 — Custom visual styling for the Streamlit interface
# =========================================================

# CSS creates the polished customer-support look.
# Streamlit widgets remain functional and accessible.

st.markdown(
    """
    <style>
    /* Main page background and spacing */
    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(20, 184, 166, 0.12),
                transparent 30%
            ),
            linear-gradient(180deg, #f7f9ff 0%, #eef4fa 100%);
        color: #172033;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Hero banner */
    .hero {
        padding: 2.1rem 2rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #172554, #3730a3 60%, #0f766e);
        box-shadow: 0 16px 40px rgba(30, 41, 99, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #99f6e4;
        margin-bottom: 0.6rem;
    }

    .hero h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.15;
        margin: 0;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 1.02rem;
        margin: 0.7rem 0 0;
    }

    /* Small status cards */
    .mini-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e0e7f0;
        border-radius: 16px;
        padding: 1rem;
        min-height: 105px;
    }

    .mini-card .number {
        color: #0f766e;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .mini-card .label {
        color: #475569;
        font-size: 0.9rem;
    }

    /* Answer cards */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.3rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
        min-height: 180px;
        margin-bottom: 0.7rem;
    }

    .answer-title {
        font-size: 1.1rem;
        font-weight: 750;
        color: #172554;
        margin-bottom: 0.25rem;
    }

    .answer-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Section headings */
    h2, h3 {
        color: #172554;
    }

    /* Text input */
    div[data-testid="stTextArea"] textarea {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        font-size: 1rem;
    }

    /* Buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        background: linear-gradient(100deg, #3730a3, #0f766e);
        color: #ffffff;
        min-height: 2.8rem;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        border: 0;
        filter: brightness(1.08);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem;
    }

    /* Streamlit alerts and expanders */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# BLOCK 8 — Render the bilingual customer-support header
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            ✦ AI CUSTOMER SUPPORT · AI வாடிக்கையாளர் உதவி
        </div>
        <h1>Jothidam Knowledge Desk</h1>
        <p>
            Your questions, answered from your Jothidam books
            and web research.
        </p>
        <p>
            உங்கள் கேள்விகளுக்கு ஜோதிட நூல்கள் மற்றும்
            இணையத் தேடல் மூலம் பதில்கள்.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# BLOCK 9 — Show the three-step workflow
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="mini-card">
            <div class="number">01 · நூல்</div>
            <b>Book Assistant</b>
            <div class="label">Searches your local Jothidam notes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="mini-card">
            <div class="number">02 · இணையம்</div>
            <b>Web Research</b>
            <div class="label">Searches online for a second answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="mini-card">
            <div class="number">03 · பதிவு</div>
            <b>Support Record</b>
            <div class="label">Saves the question and both answers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# =========================================================
# BLOCK 10 — Build the question form
# =========================================================

st.markdown("## 💬 Ask your question")
st.markdown("### உங்கள் கேள்வியைக் கேளுங்கள்")

st.caption(
    "Ask in Tamil or English. Both agents will answer each submitted question."
)

# Example questions help users explore the interface.
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    example_definition = st.button(
        "✨ What is Jothidam?",
        use_container_width=True,
    )

with example_col2:
    example_planets = st.button(
        "🪐 What are the nine planets?",
        use_container_width=True,
    )

with example_col3:
    example_rasi = st.button(
        "🌙 What is Rasi?",
        use_container_width=True,
    )

# Keep the question text and conversation history across reruns.
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# Example buttons populate the input area for the next submission.
if example_definition:
    st.session_state.question_text = "What is Jothidam?"

if example_planets:
    st.session_state.question_text = (
        "What are the nine planets mentioned in the Jothidam text?"
    )

if example_rasi:
    st.session_state.question_text = "What is Rasi in Jothidam?"

# A form prevents the app from submitting on every keystroke.
with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Your message / உங்கள் செய்தி",
        key="question_text",
        height=110,
        placeholder=(
            "Example: ஜோதிடம் என்றால் என்ன?\n"
            "Example: What is astrology?"
        ),
    )

    submitted = st.form_submit_button(
        "✦ Submit Question · கேள்வியைச் சமர்ப்பிக்கவும்",
        use_container_width=True,
    )


# =========================================================
# BLOCK 11 — Run the CrewAI workflow on submission
# =========================================================

if submitted:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning(
            "Please enter a question. / "
            "தயவுசெய்து உங்கள் கேள்வியை உள்ளிடவும்."
        )
    else:
        # The spinner gives clear feedback while all three
        # CrewAI agents execute sequentially.
        try:
            with st.spinner(
                "Our assistants are working... "
                "உங்கள் கேள்விக்குப் பதில் தயாராகிறது..."
            ):
                result = run_support_workflow(cleaned_question)

           # Show only the latest question and its answers.
           # Older interactions remain safely stored in answers_log.txt.
            st.session_state.history = [result]

            st.success(
                "Your question has been processed and logged. "
                "உங்கள் கேள்வி செயலாக்கப்பட்டு பதிவு செய்யப்பட்டது."
            )

        except Exception as error:
            # Show the error without revealing environment variables.
            st.error(
                "We couldn't process this question. "
                "Please check the terminal for technical details."
            )
            st.caption(
                "கேள்வியைச் செயலாக்க முடியவில்லை. "
                "தொழில்நுட்ப விவரங்களை Terminal-இல் பார்க்கவும்."
            )
            st.code(str(error))


# =========================================================
# BLOCK 12 — Display both agent answers in polished cards
# =========================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("## 📬 Support response")
    st.markdown("### உங்கள் கேள்விக்கான பதில்கள்")

    for index, item in enumerate(st.session_state.history):
        st.markdown(
            f"**Question / கேள்வி:** {item['question']}"
        )

        answer_col1, answer_col2 = st.columns(2, gap="large")

        # Agent 1 answer card.
        with answer_col1:
            st.markdown(
                """
                <div class="answer-card">
                    <div class="answer-title">
                        📚 Agent 1 · Jothidam Book Assistant
                    </div>
                    <div class="answer-subtitle">
                        உள்ளூர் ஜோதிட நூல்களின் அடிப்படையில்
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(item["local_answer"])

            if item["source_page"] is not None:
                st.caption(
                    f"Retrieved book passage: page {item['source_page']} "
                    f"· நூல் பக்கம்: {item['source_page']}"
                )
            else:
                st.caption(
                    "No specific source page retrieved. "
                    "குறிப்பிட்ட நூல் பக்கம் கிடைக்கவில்லை."
                )

        # Agent 2 answer card.
        with answer_col2:
            st.markdown(
                """
                <div class="answer-card">
                    <div class="answer-title">
                        🌐 Agent 2 · Web Research Assistant
                    </div>
                    <div class="answer-subtitle">
                        இணையத் தேடலின் அடிப்படையில்
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(item["web_answer"])

        # Agent 3 status and its generated record.
        with st.expander(
            "📝 Agent 3 · Support record / பதிவு விவரங்கள்"
        ):
            st.markdown(
                "The question and both answers were appended to:"
            )
            st.code(item["log_file"], language=None)

            st.markdown("**Agent 3 record:**")
            st.text(item["entry_text"])

        # Offer the recorded interaction as a downloadable text file.
        download_text = (
            f"Question / கேள்வி:\n{item['question']}\n\n"
            f"AGENT 1 — LOCAL JOTHIDAM ANSWER\n"
            f"{item['local_answer']}\n\n"
            f"AGENT 2 — WEB SEARCH ANSWER\n"
            f"{item['web_answer']}\n\n"
            f"AGENT 3 — SUPPORT RECORD\n"
            f"{item['entry_text']}\n"
        )

        st.download_button(
            label="⬇ Download this response · பதிலைப் பதிவிறக்கவும்",
            data=download_text,
            file_name=f"jothidam_response_{index + 1}.txt",
            mime="text/plain",
            key=f"download_{index}",
        )

        st.markdown("---")


# =========================================================
# BLOCK 13 — Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        Powered by CrewAI · OpenAI · Serper · Streamlit
        <br>
        உருவாக்கப்பட்டது: CrewAI · OpenAI · Serper · Streamlit
        <br><br>
        Traditional knowledge assistant for learning and research.
        <br>
        பாரம்பரிய அறிவைக் கற்றல் மற்றும் ஆய்வு செய்வதற்கான உதவியாளர்.
    </div>
    """,
    unsafe_allow_html=True,
)

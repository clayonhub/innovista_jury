"""
Jury Matching System — Fast Local Embedding Edition
===================================================
• Uses all-MiniLM-L6-v2 (384-dim, 80MB, blazing fast local inference)
• Pre-built faculty embeddings loaded from disk (run build_embeddings.py once)
• Unlimited project add / edit / delete
• Top-5 faculty matched per project + overall ranking
• Instant cosine-similarity search (NumPy vectorised)

CSV required columns:
  name, college, department, designation,
  research_interest, research_work,
  email, phone, consulting, profile_url
"""

import os
import io
import json
import hashlib
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH      = "faculty_master_list.csv"
EMB_PATH      = "faculty_minilm_embeddings.npy"
META_PATH     = "faculty_minilm_meta.json"
SESSIONS_FILE = "sessions.json"
TOP_K         = 5
EMB_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, ultra-fast 80MB model

REQUIRED_COLS = [
    "name", "college", "department", "designation",
    "research_interest", "email", "phone",
    "research_work", "consulting", "profile_url",
]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jury Matching System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — dark-mode inspired glassmorphism + vivid gradients
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Banner ────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #003399 0%, #002277 50%, #001155 100%);
    color: white;
    padding: 2.4rem 3rem 2.2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,34,119,0.45);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
}
.hero-text h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    line-height: 1.2;
}
.hero-text h1 span { color: #89b4ff; }
.hero-text p {
    margin: 0.5rem 0 0;
    font-size: 0.95rem;
    opacity: 0.75;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.hero-eyebrow {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.9;
    margin-bottom: 0.5rem;
    color: #89b4ff;
}

/* ── Metric Cards ───────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f3f0ff 100%);
    border: 1px solid #e4dcff;
    border-top: 3px solid #7c5cbf;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(124,92,191,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(124,92,191,0.15);
}
.metric-card .val {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #302b63, #7c5cbf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .lbl {
    font-size: 0.82rem;
    font-weight: 600;
    color: #7c5cbf;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ── Section Headers ────────────────────────────────────────────────── */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1a1040;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 1.2rem;
}

/* ── Project Item ───────────────────────────────────────────────────── */
.proj-item {
    background: white;
    border: 1px solid #ede9ff;
    border-left: 5px solid #6d4aff;
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 12px rgba(109,74,255,0.06);
    transition: all 0.2s;
}
.proj-item:hover {
    box-shadow: 0 6px 24px rgba(109,74,255,0.12);
    border-left-color: #4f2bfc;
}
.proj-title {
    font-weight: 700;
    color: #1a1040;
    font-size: 1rem;
}
.proj-desc {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 90%;
}

/* ── Faculty Result Card ────────────────────────────────────────────── */
.faculty-card {
    background: white;
    border: 1px solid #f0ecff;
    border-left: 5px solid #6d4aff;
    border-radius: 14px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: all 0.2s;
}
.faculty-card:hover {
    box-shadow: 0 8px 28px rgba(109,74,255,0.13);
    transform: translateX(2px);
}
.faculty-card.gold   { border-left-color: #f0b429; background: linear-gradient(to right, #fffdf0, #fff); }
.faculty-card.silver { border-left-color: #94a3b8; }
.faculty-card.bronze { border-left-color: #cd7f32; }

.fac-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1040;
}

.score-pill {
    display: inline-block;
    background: linear-gradient(135deg, #6d4aff, #4f2bfc);
    color: white;
    border-radius: 30px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    float: right;
}
.gold   .score-pill { background: linear-gradient(135deg, #f0b429, #d97706); }
.silver .score-pill { background: linear-gradient(135deg, #94a3b8, #64748b); }
.bronze .score-pill { background: linear-gradient(135deg, #cd7f32, #9a5f1e); }

.badge {
    display: inline-block;
    background: #f0ecff;
    color: #4f2bfc;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 5px;
    margin-top: 5px;
}

/* ── Project Header in Results ──────────────────────────────────────── */
.proj-result-header {
    background: linear-gradient(90deg, #302b63, #6d4aff);
    color: white;
    border-radius: 12px;
    padding: 0.85rem 1.4rem;
    margin: 1.8rem 0 1rem;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: -0.2px;
}

/* ── Overall Card ───────────────────────────────────────────────────── */
.overall-card {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 1px solid #c8e6c9;
    border-left: 5px solid #43a047;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 10px rgba(67,160,71,0.07);
}
.overall-card .score-pill {
    background: linear-gradient(135deg, #43a047, #2e7d32);
}

/* ── Divider ────────────────────────────────────────────────────────── */
.custom-divider {
    border: none;
    border-top: 2px solid #f3f0ff;
    margin: 2rem 0;
}

/* ── Streamlit overrides ────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 10px !important;
    border-color: #e0d8ff !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6d4aff !important;
    box-shadow: 0 0 0 3px rgba(109,74,255,0.15) !important;
}
.stExpander { border-radius: 12px !important; border-color: #ede9ff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MiniLM model (fast)…")
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faculty_df(csv_path: str, _mtime: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=lambda c: c in REQUIRED_COLS)
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    df = df.dropna(subset=["name"]).fillna("").reset_index(drop=True)
    return df


def build_doc_text(row) -> str:
    """Combine research_interest + research_work into one embedding document."""
    ri   = str(row.get("research_interest", "")).strip()
    rw   = str(row.get("research_work", "")).strip()
    name = str(row.get("name", "")).strip()
    dept = str(row.get("department", "")).strip()
    parts = [f"Faculty: {name}", f"Department: {dept}"]
    if ri:
        parts.append(f"Research Interests: {ri}")
    if rw:
        parts.append(f"Research Work: {rw[:600]}")
    return "\n".join(parts)


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(
            df[["name", "research_interest", "research_work"]], index=False
        ).values.tobytes()
    ).hexdigest()[:16]


@st.cache_data(show_spinner="Loading pre-built faculty embeddings…")
def load_embeddings_from_disk(csv_hash: str, n_rows: int):
    """
    Load pre-built embeddings from disk.
    Returns (emb_matrix float32 L2-normed, meta list[dict])
    Raises FileNotFoundError if files are missing or stale.
    """
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("missing")

    with open(META_PATH, encoding="utf-8") as f:
        stored = json.load(f)

    if stored.get("csv_hash") != csv_hash or stored.get("n_rows") != n_rows:
        raise FileNotFoundError("stale")

    emb   = np.load(EMB_PATH).astype(np.float32)
    meta  = stored["rows"]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms, meta


def cosine_topk(emb_matrix: np.ndarray, query_vec: np.ndarray, k: int):
    """Fast cosine similarity top-k via matrix dot product (vecs are L2-normed)."""
    sims = emb_matrix @ query_vec
    k    = min(k, len(sims))
    idx  = np.argpartition(sims, -k)[-k:]
    idx  = idx[np.argsort(sims[idx])[::-1]]
    return idx, sims[idx]


def embed_query(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a project query directly via local model + L2 normalise."""
    vec = model.encode(
        text, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGER (Disk Persistence)
# ─────────────────────────────────────────────────────────────────────────────
def load_all_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_all_sessions(data: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_current_session():
    """Save the current user's projects and results to disk."""
    user = st.session_state.username
    if not user: return
    
    sessions = load_all_sessions()
    if user not in sessions:
        sessions[user] = {"created": time.time()}
    
    sessions[user]["last_active"] = time.time()
    sessions[user]["projects"]    = st.session_state.projects
    sessions[user]["results"]     = st.session_state.results
    save_all_sessions(sessions)

def load_user_session(username: str):
    """Load projects/results for a user into session_state."""
    sessions = load_all_sessions()
    data = sessions.get(username, {})
    st.session_state.projects = data.get("projects", [])
    st.session_state.results  = data.get("results", None)
    st.session_state.username = username
    save_current_session() # Update last_active and save if new


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "username":  None,
        "projects":  [],
        "proj_key":  0,
        "results":   None,
        "edit_idx":  None,
        "show_form": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — CSV + Embeddings (model loaded separately for query embedding)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    st.error(f"❌ `{CSV_PATH}` not found. Place your faculty CSV in the same folder as this app.")
    st.stop()

csv_mtime  = os.path.getmtime(CSV_PATH)
faculty_df = load_faculty_df(CSV_PATH, csv_mtime)
csv_hash   = _df_hash(faculty_df)

try:
    emb_matrix, meta_rows = load_embeddings_from_disk(csv_hash, len(faculty_df))
except FileNotFoundError as _e:
    reason = "stale (CSV has changed)" if "stale" in str(_e) else "not found"
    st.error(
        f"❌ Pre-built embeddings are {reason}.\n\n"
        "**Run this command once in your terminal, then restart Streamlit:**\n\n"
        "```\npython build_embeddings.py\n```"
    )
    st.stop()

# Load the small model for runtime querying
model = get_model()


# ─────────────────────────────────────────────────────────────────────────────
# ── LOGIN SCREEN ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.username:
    st.markdown(f"""
    <div class="hero" style="text-align:center; flex-direction:column; padding: 4rem 2rem;">
      <div class="hero-eyebrow">Tata Innovista</div>
      <h1 style="font-size:2.2rem; margin-top:0.5rem; letter-spacing:1px; line-height:1.2;">Jury <span>Matching</span> System</h1>
      <p style="opacity:0.8; font-size:1.1rem; max-width:600px; margin: 1rem auto;">
        Sign in to your workspace. Your projects and matching results are saved automatically.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="section-title">Login / Select Workspace</p>', unsafe_allow_html=True)
        
        all_sess = load_all_sessions()
        
        # Determine if we can create new sessions
        can_create = len(all_sess) < 3
        
        with st.form("login_form"):
            new_user = st.text_input("Enter Workspace Name", placeholder="e.g. Alice's Projects")
            sub = st.form_submit_button("Enter Workspace", type="primary", use_container_width=True)
            if sub:
                uname = new_user.strip()
                if not uname:
                    st.error("Please enter a name.")
                elif uname not in all_sess and not can_create:
                    st.error("Maximum 3 workspaces reached. Please delete an existing one or use an existing name.")
                else:
                    load_user_session(uname)
                    st.rerun()
        
        if all_sess:
            st.markdown("<br><p class='section-sub'><b>Or resume an existing workspace:</b></p>", unsafe_allow_html=True)
            for name in list(all_sess.keys()):
                c_lbl, c_del = st.columns([4, 1])
                with c_lbl:
                    if st.button(f"📂 {name}", key=f"join_{name}", use_container_width=True):
                        load_user_session(name)
                        st.rerun()
                with c_del:
                    if st.button("Delete", key=f"del_sess_{name}"):
                        del all_sess[name]
                        save_all_sessions(all_sess)
                        st.rerun()
            
            st.caption(f"{len(all_sess)} / 3 workspaces used.")
        
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ── HERO BANNER (LOGGED IN) ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
cc1, cc2 = st.columns([8, 2])
with cc1:
    st.markdown(f"""
    <div class="hero" style="padding: 1.8rem 2.5rem; margin-bottom: 1rem;">
      <div class="hero-text">
        <div class="hero-eyebrow">Tata Innovista</div>
        <h1 style="font-size: 1.5rem;">Jury <span>Matching</span> System</h1>
      </div>
    </div>
    """, unsafe_allow_html=True)
with cc2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Workspace:**<br>`{st.session_state.username}`", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.username = None
        st.session_state.projects = []
        st.session_state.results  = None
        st.rerun()

# ── Stats Row ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
for col, val, lbl in [
    (c1, f"{len(faculty_df):,}", "Faculty Indexed"),
    (c2, str(len(st.session_state.projects)), "Projects Added"),
    (c3, str(TOP_K), "Top Matches / Project"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION A — Browse Faculty ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Browse Faculty Database", expanded=False):
    b1, b2, b3 = st.columns(3)
    with b1:
        q_name = st.text_input("Search Name", key="br_name", placeholder="e.g. Sharma")
    with b2:
        q_coll = st.text_input("Search College", key="br_coll", placeholder="e.g. IIT Kanpur")
    with b3:
        q_dept = st.text_input("Search Dept", key="br_dept", placeholder="e.g. Computer Science")

    mask = pd.Series([True] * len(faculty_df))
    if q_name.strip():
        mask &= faculty_df["name"].str.contains(q_name.strip(), case=False, na=False)
    if q_coll.strip():
        mask &= faculty_df["college"].str.contains(q_coll.strip(), case=False, na=False)
    if q_dept.strip():
        mask &= faculty_df["department"].str.contains(q_dept.strip(), case=False, na=False)

    filtered  = faculty_df[mask]
    st.caption(f"Showing **{len(filtered):,}** of **{len(faculty_df):,}** faculty")

    disp_cols = [c for c in
                 ["name", "designation", "college", "department", "research_interest", "email", "profile_url"]
                 if c in filtered.columns]
    df_disp = filtered[disp_cols].rename(columns={
        "name": "Name", "designation": "Designation", "college": "College",
        "department": "Department", "research_interest": "Research Interests",
        "email": "Email", "profile_url": "Profile URL",
    })
    col_cfg = {}
    if "Profile URL" in df_disp.columns:
        col_cfg["Profile URL"] = st.column_config.LinkColumn(label="Profile URL", display_text="Open ↗")
    st.dataframe(df_disp, use_container_width=True, height=280, column_config=col_cfg)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION B — Project Manager ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-title">Project Manager</p>'
    '<p class="section-sub">Add, edit, or remove projects. Each will be matched against all faculty.</p>',
    unsafe_allow_html=True,
)

editing    = st.session_state.edit_idx is not None
form_label = "Edit Project" if editing else "Add New Project"
form_open  = editing or st.session_state.show_form

with st.expander(form_label, expanded=form_open):
    default_title = st.session_state.projects[st.session_state.edit_idx]["title"] if editing else ""
    default_desc  = st.session_state.projects[st.session_state.edit_idx]["description"] if editing else ""

    with st.form(key=f"proj_form_{st.session_state.proj_key}", clear_on_submit=True):
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            p_title = st.text_input("Project Title *", value=default_title,
                                    placeholder="e.g. AI for Medical Imaging")
        with fc2:
            p_desc = st.text_area("Project Description *", value=default_desc, height=110,
                                  placeholder="Describe domain, goals, and technologies (100–200 words for best accuracy)")

        s1, s2, _ = st.columns([2, 2, 6])
        with s1:
            submit_btn = st.form_submit_button(
                "Save Changes" if editing else "Add Project",
                type="primary", use_container_width=True,
            )
        with s2:
            cancel_btn = st.form_submit_button("Cancel", use_container_width=True)

        if submit_btn:
            if not p_title.strip() or not p_desc.strip():
                st.error("Both Title and Description are required.")
            else:
                proj = {"title": p_title.strip(), "description": p_desc.strip()}
                if editing:
                    st.session_state.projects[st.session_state.edit_idx] = proj
                    st.session_state.edit_idx = None
                else:
                    st.session_state.projects.append(proj)
                    st.session_state.show_form = False
                st.session_state.results  = None
                st.session_state.proj_key += 1
                save_current_session()
                st.rerun()

        if cancel_btn:
            st.session_state.edit_idx  = None
            st.session_state.show_form = False
            st.session_state.proj_key += 1
            st.rerun()

if not editing and not st.session_state.show_form:
    if st.button("+ Add Project", key="open_add_form"):
        st.session_state.show_form = True
        st.rerun()

if st.session_state.projects:
    st.markdown(f"**{len(st.session_state.projects)} project(s):**")
    for i, p in enumerate(st.session_state.projects):
        c_main, c_edit, c_del = st.columns([10, 1, 1])
        with c_main:
            st.markdown(f"""
<div class="proj-item">
  <div class="proj-title"><i class="fa-regular fa-folder" style="margin-right:8px;opacity:0.7;"></i>{p['title']}</div>
  <div class="proj-desc">{p['description'][:180]}{'…' if len(p['description']) > 180 else ''}</div>
</div>
""", unsafe_allow_html=True)
        with c_edit:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Edit", key=f"edit_{i}", help="Edit project"):
                st.session_state.edit_idx  = i
                st.session_state.show_form = False
                st.rerun()
        with c_del:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Delete", key=f"del_{i}", help="Remove project"):
                st.session_state.projects.pop(i)
                if st.session_state.edit_idx == i:
                    st.session_state.edit_idx = None
                st.session_state.results = None
                save_current_session()
                st.rerun()
else:
    st.info("No projects yet — click **+ Add Project** above to get started.")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION C — Run Matching ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Run Matching</p>', unsafe_allow_html=True)

n_proj = len(st.session_state.projects)
col_run, col_clr = st.columns([2, 1])
with col_run:
    run_btn = st.button(
        f"Match Faculty to {n_proj} Project{'s' if n_proj != 1 else ''}",
        type="primary", disabled=(n_proj == 0), use_container_width=True,
    )
with col_clr:
    if st.button("Clear All", use_container_width=True):
        st.session_state.projects = []
        st.session_state.results  = None
        save_current_session()
        st.rerun()

if run_btn and st.session_state.projects:
    prog_match  = st.progress(0, "Running semantic matching…")
    project_res = []
    jury_scores: dict = {}

    for pi, p in enumerate(st.session_state.projects):
        prog_match.progress(
            int(pi / n_proj * 90),
            f"Matching project {pi + 1}/{n_proj}: {p['title'][:40]}…"
        )
        query  = f"{p['title']}: {p['description']}"
        q_vec  = embed_query(query, model)
        top_idx, top_sims = cosine_topk(emb_matrix, q_vec, TOP_K)

        rankings = []
        for idx, sim in zip(top_idx, top_sims):
            row  = meta_rows[int(idx)]
            name = row.get("name", "Unknown")
            rankings.append({**row, "similarity": round(float(sim), 4)})
            jury_scores[name] = jury_scores.get(name, 0.0) + float(sim)

        project_res.append({"project": p["title"], "rankings": rankings})

    prog_match.progress(100, "Done!")
    time.sleep(0.3)
    prog_match.empty()

    st.session_state.results = {
        "project_results":   project_res,
        "jury_total_scores": jury_scores,
        "n_projects":        n_proj,
    }
    save_current_session()


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION D — Results ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.results:
    res        = st.session_state.results
    proj_res   = res["project_results"]
    score_map  = res["jury_total_scores"]
    n_proj_res = res["n_projects"]

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Matching Results</p>', unsafe_allow_html=True)

    card_cls = {1: "gold", 2: "silver", 3: "bronze"}
    rank_ico = {1: "#1", 2: "#2", 3: "#3"}

    for pr in proj_res:
        st.markdown(
            f'<div class="proj-result-header"><i class="fa-solid fa-diagram-project" style="margin-right:8px;"></i>Project: {pr["project"]}</div>',
            unsafe_allow_html=True,
        )

        for rank_i, r in enumerate(pr["rankings"], 1):
            cls   = card_cls.get(rank_i, "")
            icon  = rank_ico.get(rank_i, f"#{rank_i}")
            name  = r.get("name", "")
            url   = r.get("profile_url", "").strip()
            desig = r.get("designation", "").strip()
            coll  = r.get("college", "").strip()
            dept  = r.get("department", "").strip()
            email = r.get("email", "").strip()
            ri    = r.get("research_interest", "").strip()
            score = r["similarity"]

            name_html = (
                f'<a href="{url}" target="_blank"'
                f' style="color:#4f2bfc;text-decoration:underline;font-weight:700;">{name}</a>'
                if url else f'<strong style="color:#1a1040;">{name}</strong>'
            )
            ri_snippet   = ri[:160] + "…" if len(ri) > 160 else ri
            email_html   = f'<span class="badge"><i class="fa-regular fa-envelope" style="margin-right:4px;"></i>{email}</span>' if email else ""
            ri_html      = (
                f'<div style="font-size:0.83rem;color:#555;margin-top:6px;font-style:italic;">'
                f'{ri_snippet}</div>'
            ) if ri_snippet else ""
            profile_btn  = (
                f'<a href="{url}" target="_blank" style="'
                'display:inline-flex;align-items:center;gap:6px;'
                'background:linear-gradient(135deg,#003399,#0055cc);'
                'color:white;border-radius:8px;padding:5px 16px;'
                'font-size:0.8rem;font-weight:600;text-decoration:none;'
                'margin-top:8px;box-shadow:0 2px 8px rgba(0,51,153,0.3);">'
                '<i class="fa-solid fa-arrow-up-right-from-square"></i> View Profile</a>'
            ) if url else ""

            desig_badge = f"<span class='badge'>{desig}</span>" if desig else ""

            st.markdown(f"""
<div class="faculty-card {cls}">
  <div>
    <span style="font-weight:800;font-size:0.85rem;color:#6d4aff;margin-right:6px;">{icon}</span>
    <span class="fac-name">{name_html}</span>
    <span class="score-pill">Score: {score:.4f}</span>
  </div>
  <div style="margin-top:5px;">
    <span class="badge">{coll}</span>
    <span class="badge">{dept}</span>
    {desig_badge}
    {email_html}
  </div>
  {ri_html}
  {profile_btn}
</div>
""", unsafe_allow_html=True)

    # ── Overall Ranking ──────────────────────────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Overall Top Faculty</p>'
        '<p class="section-sub">Ranked by cumulative match score across all submitted projects.</p>',
        unsafe_allow_html=True,
    )

    avg_scores = sorted(
        [(nm, tot / n_proj_res) for nm, tot in score_map.items()],
        key=lambda x: x[1], reverse=True,
    )

    all_result_rows: dict = {}
    for pr in proj_res:
        for r in pr["rankings"]:
            all_result_rows.setdefault(r["name"], r)

    for rank_i, (nm, avg) in enumerate(avg_scores[:TOP_K], 1):
        cls   = card_cls.get(rank_i, "")
        icon  = rank_ico.get(rank_i, f"#{rank_i}")
        m     = all_result_rows.get(nm, {})
        url   = m.get("profile_url", "").strip()
        coll  = m.get("college", "").strip()
        dept  = m.get("department", "").strip()
        desig = m.get("designation", "").strip()
        ri    = m.get("research_interest", "").strip()
        ri_snippet = ri[:160] + "…" if len(ri) > 160 else ri

        name_html_ov = (
            f'<a href="{url}" target="_blank"'
            f' style="color:#1b5e20;text-decoration:underline;font-weight:700;">{nm}</a>'
            if url else f'<strong style="color:#1a1040;">{nm}</strong>'
        )
        ri_html_ov = (
            f'<div style="font-size:0.83rem;color:#555;margin-top:6px;font-style:italic;">'
            f'{ri_snippet}</div>'
        ) if ri_snippet else ""
        profile_btn_ov = (
            f'<a href="{url}" target="_blank" style="'
            'display:inline-flex;align-items:center;gap:6px;'
            'background:linear-gradient(135deg,#2e7d32,#1b5e20);'
            'color:white;border-radius:8px;padding:5px 16px;'
            'font-size:0.8rem;font-weight:600;text-decoration:none;'
            'margin-top:8px;box-shadow:0 2px 8px rgba(46,125,50,0.3);">'
            '<i class="fa-solid fa-arrow-up-right-from-square"></i> View Profile</a>'
        ) if url else ""
        desig_badge_ov = (
            f"<span class='badge' style='background:#c8e6c9;color:#2e7d32;'>{desig}</span>"
        ) if desig else ""

        st.markdown(f"""
<div class="overall-card">
  <div>
    <span style="font-weight:800;font-size:0.85rem;color:#2e7d32;margin-right:6px;">{icon}</span>
    <span class="fac-name">{name_html_ov}</span>
    <span class="score-pill">Avg: {avg:.4f}</span>
  </div>
  <div style="margin-top:5px;">
    <span class="badge" style="background:#c8e6c9;color:#2e7d32;">{coll}</span>
    <span class="badge" style="background:#c8e6c9;color:#2e7d32;">{dept}</span>
    {desig_badge_ov}
  </div>
  {ri_html_ov}
  {profile_btn_ov}
</div>
""", unsafe_allow_html=True)

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Export Results</p>', unsafe_allow_html=True)

    score_lookup: dict = {}
    for pr in proj_res:
        for r in pr["rankings"]:
            score_lookup.setdefault(r["name"], {})[pr["project"]] = r["similarity"]

    proj_titles = [pr["project"] for pr in proj_res]

    export_rows = []
    for rank_i, (nm, avg) in enumerate(avg_scores, 1):
        m   = all_result_rows.get(nm, {})
        row = {
            "Rank":         rank_i,
            "Faculty Name": nm,
            "Designation":  m.get("designation", ""),
            "College":      m.get("college", ""),
            "Department":   m.get("department", ""),
            "Email":        m.get("email", ""),
            "Profile URL":  m.get("profile_url", ""),
            "Consulting":   m.get("consulting", ""),
        }
        for pt in proj_titles:
            row[pt] = score_lookup.get(nm, {}).get(pt, 0.0)
        row["Overall Avg Score"] = round(avg, 4)
        export_rows.append(row)

    export_df = pd.DataFrame(export_rows)
    st.dataframe(
        export_df, use_container_width=True, height=300,
        column_config={"Profile URL": st.column_config.LinkColumn(display_text="Open ↗")},
    )

    ex1, ex2 = st.columns(2)
    with ex1:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
            export_df.to_excel(w, index=False, sheet_name="Jury Matching")
        xbuf.seek(0)
        st.download_button(
            "⬇️ Download Excel (.xlsx)", data=xbuf,
            file_name="jury_matching_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "⬇️ Download CSV (.csv)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="jury_matching_results.csv",
            mime="text/csv", use_container_width=True,
        )

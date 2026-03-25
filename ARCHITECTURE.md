# Jury Matching System: Architecture & Technical Design Document

This document provides an in-depth explanation of the Jury Matching System's architecture, its operational workflow, the tools utilized, design choices, comparisons against alternatives, and considerations for future scalability and accuracy optimizations.

---

## 1. System Overview

The **Jury Matching System** is an AI-powered application designed to automate and optimize the process of assigning faculty members (Jury) to specific academic or research projects. Instead of relying on manual keyword searches, the system uses **Natural Language Processing (NLP)** and **Semantic Vector Search** to understand the context and meaning of a project's description and match it against the research interests and past work of faculty members.

## 2. How the System Works

The workflow of the system is divided into two phases: **Pre-computation (Offline)** and **Inference (Online/Runtime)**.

### Phase 1: Pre-computation (Data Ingestion)
1. **Data Source:** Faculty data is maintained in a CSV file (`faculty_master_list.csv`) containing columns like name, department, research interests, and previous research works.
2. **Chunking Document Construction:** The system breaks down massive continuous blocks of text (like 40+ paper citations) into individual, isolated "chunks" using `|` delimiters or overlapping sliding sentence-windows.
3. **Embedding Generation:** A side-script (`build_embeddings.py`) passes these thousands of text chunks through an Embedding Model to generate a high-dimensional numerical representation (a "Vector") of the text.
4. **Storage:** The generated vectors are saved locally to disk as a NumPy array (`faculty_bge_embeddings.npy`) along with a metadata layout map (`faculty_bge_chunk_map.npy`) and JSON file (`faculty_bge_meta.json`).

### Phase 2: Runtime Inference
1. **User Workspaces:** A user logs into the Streamlit web interface using a specific workspace name. Workspaces are persisted locally (`sessions.json`).
2. **Project Input:** The user adds project details (Title and Description).
3. **Query Embedding:** The application takes the project's title and description and passes it through the same local Embedding Model, generating a "Query Vector".
4. **Vector Search (Cosine Similarity):** The application performs a rapid mathematical calculation—a dot product between the Project Query Vector and the entire matrix of Faculty Vectors. Since vectors are clustered by semantic meaning, this calculates the distance (similarity) between the project and all faculties.
5. **Ranking:** The system pulls the Top-K (default top 5) highest scoring faculty members and presents them in a detailed UI, ranked from Gold to Bronze contextually.

---

## 3. Tools Used & Their Functions

| Tool / Library | Function in the System |
| :--- | :--- |
| **Streamlit** | The frontend web framework. Used to quickly build a highly interactive, responsive, and aesthetically pleasing single-page application (SPA) with no raw HTML/JS required. |
| **Sentence-Transformers** | Python NLP library used to load and run the pre-trained embedding model. |
| **BAAI/bge-large-en-v1.5** | The specific Machine Learning model. It takes sentences/paragraphs and maps them to a massive 1024-dimensional dense vector space. This is a state-of-the-art open-source embedding model. |
| **NumPy** | High-performance scientific computing library in Python. Used for storing the embedding matrix and executing hyper-fast vectorized mathematical aggregation operations (`maximum.at`). |
| **Pandas** | Data manipulation library used to load, clean, and structure the raw `faculty_master_list.csv` data before displaying or embedding. |
| **JSON** | Standard file format used to persistently store user workspaces, sessions, and project data on the local disk. |

---

## 4. Why Was This Approach Chosen?

This specific architecture—**Local "Chunk-Max" Embeddings with NumPy Vectorization**—was engineered to solve critical bottlenecks in standard semantic search:

1. **Solving "Dense Vector Dilution" (Chunk-Max Architecture):** Traditional searches pass a professor's entire resume to the AI at once. The AI mathematically averages (Mean Pools) the vector. If 39 past papers are irrelevant and 1 is a perfect match, the perfect match is diluted down to 2% weight and the professor disappears from rankings.
   * **Our Solution:** We systematically break the text down into isolated chunks. The AI generates independent vectors for every single paper. During search, the software evaluates all chunks and assigns the faculty the **maximum** score of any single chunk (`np.maximum.at`), bypassing dilution entirely and perfectly hunting specialized topics.
   
2. **Transitioning from MiniLM to BGE-Large (Comprehension Power):** 
   * **Inception (all-MiniLM-L6-v2):** Originally chosen because it is 80MB and calculates extremely fast on basic CPUs. However, it only creates 384-dimensional vectors and struggles to comprehend highly specialized academic physics and metallurgy terms.
   * **Now (BAAI/bge-large-en-v1.5):** Upgraded to a 1.3 Gigabyte, 1024-dimensional model. While it requires a GPU (like Google Colab) to process the initial ~20,000 document chunks, running inference locally on a small query takes only ~2 seconds. The result is total comprehension of the heaviest academic terminology.
   
3. **No Heavy Database Infrastructure:** We deliberately avoided setting up Vector Databases like Pinecone or Qdrant. A purely self-contained, offline application using bare NumPy operations provides extreme security for proprietary project details while requiring zero paid API keys.

---

## 5. What Could Have Been the Alternatives?

### Alternative 1: Traditional Keyword Matching (TF-IDF or BM25)
* **How it works:** Searching for exact word matches (e.g., matching the word "Neural" in a project to "Neural" in a faculty profile).
* **Why it was rejected:** Lexical search fails to understand synonyms and semantics. A project about "Deep Learning for Vision" would fail to match a faculty whose profile says "Convolutional Networks for Image processing", even though they are exactly the same domain.

### Alternative 2: Commercial API LLM Embeddings (e.g., OpenAI `text-embedding-3-small`)
* **How it works:** Sending the text to OpenAI to generate vectors.
* **Why it was rejected:** Introduces latency (network round-trips), requires recurring payments, and risks data privacy if the projects encompass sensitive intellectual property. 

### Alternative 3: Vector Databases (ChromaDB, Pinecone, FAISS)
* **How it works:** Storing the embeddings in a specialized database optimized for Approximate Nearest Neighbor (ANN) search.
* **Why it was rejected:** Unnecessary overhead. Vector databases shine when you are querying millions of embeddings. For a dataset of hundreds or thousands of faculty members, an exhaustive search (dot product) in RAM is actually faster and far simpler to deploy.

---

## 6. How Can We Optimize Scoring So We NEVER Miss Anything?

The current architecture elegantly solves the semantic matching problem. Dense Vectors (BGE-large) are phenomenal for understanding **concepts** and matching "vibes" (e.g., matching "automobile" to "car"). 
However, Dense Vectors occasionally struggle to execute exact, specialized keyword lookups (e.g., hunting for the exact term "Ni-Co-SiC Nanocomposite"). 

If the absolute priority for the Jury System is that it **never misses a specialized faculty member**, the architectural gold-standard is **Hybrid Search (Dense + Sparse)**:

### 1. Dense Search (What we have now)
Uses the AI model (`BAAI/bge-large`) to evaluate deep semantic meaning and conceptual overlap. It grasps the abstract themes of the project and finds faculty researching those abstract physics principles.

### 2. Sparse Search (The Next Step: BM25 / TF-IDF)
This is an incredibly fast, traditional mathematical algorithm that ignores semantic meaning entirely and functions strictly to count exact, rare keyword occurrences. It acts as a laser-guided missile. If a project asks for a specific proprietary term, BM25 scans the chunks, isolates the faculty possessing that term, and emits a massive mathematical spike.

### 3. Reciprocal Rank Fusion (RRF Workflow)
1. The software executes the Semantic (Dense) search and scores everyone 0.0 to 1.0.
2. The software executes the Exact-Keyword (Sparse) search and scores everyone.
3. The system blends the two scores (Alpha Blending/RRF). 

A Hybrid Search architecture guarantees that the engine inherently comprehends broad, complex physics concepts while simultaneously acting as a foolproof keyword-hunting tool, resulting in mathematically flawless jury mapping.

### C. Scaling for Massive Datasets (Millions of rows)
If the system scales beyond local university faculty lists to encompass millions of global researchers:
* **Alternative:** The NumPy dot product would become too slow. At this scale, transitioning to **FAISS (Facebook AI Similarity Search)** or a managed Vector Database (Pinecone) would become mandatory to retrieve results efficiently using Approximate Nearest Neighbors (ANN) rather than exact mathematical comparisons.

### Conclusion
The current architecture elegantly solves the semantic matching problem using an uncompromised blend of lightning-fast local CPU inference and in-memory mathematical evaluation, ensuring zero downtime, strict privacy, and excellent baseline accuracy for academic environments.

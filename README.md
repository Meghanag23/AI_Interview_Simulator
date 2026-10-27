# RAG-Enhanced Autonomous AI Interview Simulator

## 1. Project Overview

This project is a RAG-enhanced autonomous AI interview simulator for behavioral interview practice. The system retrieves role-specific behavioral interview questions, evaluates candidate answers using the STAR framework, adapts difficulty based on performance, and generates a personalized final report.

It uses Retrieval-Augmented Generation for question retrieval, Gemini for STAR-based answer evaluation, and FastAPI for the backend demo.

## 2. Main Features

1. Retrieves relevant behavioral interview questions using dense semantic retrieval.
2. Uses SentenceTransformer embeddings and ChromaDB for vector search.
3. Implements BM25 keyword retrieval as a baseline comparison.
4. Evaluates candidate answers using STAR: Situation, Task, Action, and Result.
5. Adapts question difficulty based on candidate performance.
6. Tracks session memory, weak topics, weak STAR components, and scores.
7. Generates a final personalized interview performance report.
8. Provides a working FastAPI Swagger UI demo.

## 3. Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- SentenceTransformer
- ChromaDB
- rank-bm25
- Gemini API
- JSON dataset

## 4. Folder Structure

```text
RAG_AI_Interview_Simulator_Final/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   └── questions.json
│
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── retriever.py
│   ├── evaluator.py
│   ├── session.py
│   ├── agent.py
│   └── experiment.py
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── screenshots/
│   └── demo screenshots
│
└── results/
    └── experiment_results.txt
```

## 5. Folder Structure

This project was developed and tested using Python 3.11. Python 3.11 is recommended because some dependencies such as ChromaDB and pydantic may cause installation issues with newer Python versions.

### 5.1 Open the Project Folder

Open the terminal and go to the project folder.

### 5.2 Create a Virtual Environment

Create a new virtual environment using Python 3.11.

bash python3.11 -m venv venv 

This creates a separate Python environment for the project so that the required packages do not conflict with other Python projects.

### 5.3 Activate the Virtual Environment

For Mac or Linux:

bash source venv/bin/activate 

For Windows:

bash venv\Scripts\activate 

After activation, the terminal should show (venv) at the beginning of the command line.

### 5.4 Upgrade pip

Before installing the dependencies, upgrade pip.

bash python -m pip install --upgrade pip 

### 5.5 Install Required Dependencies

Install all required Python packages using requirements.txt.

bash pip install -r requirements.txt 

The main dependencies used in this project are:

text fastapi uvicorn python-dotenv sentence-transformers chromadb rank-bm25 google-generativeai pydantic numpy scikit-learn 

## 6. API Key Setup

This project uses the Gemini API for STAR-based answer evaluation. 

Create a .env file in the root project folder.

bash cp .env.example .env 

Open the .env file and add your Gemini API key.

env GEMINI_API_KEY=your_gemini_api_key_here 

The submitted project includes .env.example, but it does not include the real .env file.

Example .env.example file:

env GEMINI_API_KEY=your_gemini_api_key_here 

## 7. Build the Question Database and Retrieval Indexes

Before running the API, build the ChromaDB vector index and BM25 baseline index.

bash python -m src.database 

Expected output may look similar to this:

text Loaded questions from data/questions.json SentenceTransformer model loaded ChromaDB collection created Questions embedded and stored successfully BM25 index prepared Database build complete 

Note: The first run may take longer because the SentenceTransformer model may download automatically.

## 8. Run the Retrieval Experiment

To reproduce the retrieval comparison between RAG dense retrieval and BM25 keyword retrieval, run:

bash python -m src.experiment 

Expected output:

text Total queries: 5  RAG Results: Mean Precision@3: 0.40 Mean MRR: 1.00 Average latency: 164.3 ms  BM25 Results: Mean Precision@3: 0.2666 Mean MRR: 0.80 Average latency: approximately 0 ms 

These results show that RAG dense retrieval performed better than BM25 keyword retrieval for the manually labeled interview retrieval queries.

## 9. Start the FastAPI Server

Run the FastAPI backend using Uvicorn.

bash uvicorn api.main:app --reload --port 8001 

Expected output:

text Uvicorn running on http://127.0.0.1:8001 Application startup complete 

If port 8001 is already in use, run the server on another port.

bash uvicorn api.main:app --reload --port 8002 

## 10. Open Swagger UI

After the server starts, open the following URL in a browser.

text http://127.0.0.1:8001/docs 

Swagger UI can be used to test all project endpoints.

text GET /health POST /session/start POST /session/answer GET /session/{session_id}/report 

## 11. API Endpoints

### 11.1 Health Check

Endpoint:

text GET /health 

Purpose:

This endpoint checks whether the FastAPI server is running.

Example response:

json {   "status": "ok" } 

### 11.2 Start Interview Session

Endpoint:

text POST /session/start 

Purpose:

This endpoint starts a new interview session. The user provides the candidate role, target company, and focus areas. The system returns a session ID and the first RAG-selected behavioral interview question.

Example request:

json {   "candidate_role": "Backend SDE",   "target_company": "Amazon",   "focus_areas": ["leadership", "delivery"] } 

Example response:

json {   "session_id": "abc123",   "message": "Welcome to your AI Interview Simulator session.\n\nRole: Backend SDE\nCompany: Amazon\nCurrent difficulty: medium\n\nQuestion 1:\nTell me about a time you delivered a project under tight deadlines or with limited resources." } 

### 11.3 Submit Candidate Answer

Endpoint:

text POST /session/answer 

Purpose:

This endpoint submits the candidate answer for evaluation. The system evaluates the answer using the STAR framework, updates session memory, adapts difficulty, and returns the next question.

Example request:

json {   "session_id": "abc123",   "candidate_answer": "At my previous company, two engineers disagreed on the implementation approach. I listened to both sides, compared the tradeoffs, proposed a middle-ground design, and helped the team move forward." } 

Example response:

json {   "message": "STAR Evaluation:\nSituation: 7/10\nTask: 7/10\nAction: 6/10\nResult: 5/10\nOverall: 6.1/10\n\nFeedback: The answer explains the situation and task, but the result should include a clearer measurable outcome.\nWeakest component: result\n\nUpdated difficulty: medium\nQuestions completed: 1/6\n\nNext Question 2:\nTell me about a time you handled competing priorities." } 

### 11.4 Generate Final Report

Endpoint:

text GET /session/{session_id}/report 

Purpose:

This endpoint generates the final session report. The report includes the candidate role, target company, average score, final difficulty, topic scores, weakest topic, weakest STAR component, and recommendations.

Example response:

json {   "session_id": "abc123",   "candidate_role": "Backend SDE",   "target_company": "Amazon",   "total_questions": 3,   "average_score": 7.22,   "final_difficulty": "hard",   "scores_by_topic": {     "delivery": 6.1,     "leadership": 7.35,     "ownership": 8.2   },   "weakest_topic": "delivery",   "most_common_weak_component": "action",   "recommendations": [     "Practice more delivery questions.",     "Give more detail on the specific steps you personally took.",     "Include measurable results when describing outcomes."   ] } 

## 12. Demo Steps Using Swagger UI

Follow these steps to run the project demo.

### Step 1: Start the FastAPI server

bash uvicorn api.main:app --reload --port 8001 

### Step 2: Open Swagger UI

text http://127.0.0.1:8001/docs 

### Step 3: Run POST /session/start

Use this sample request:

json {   "candidate_role": "Backend SDE",   "target_company": "Amazon",   "focus_areas": ["leadership", "delivery"] } 

### Step 4: Copy the returned session_id

The session_id is needed for submitting answers and generating the report.

### Step 5: Run POST /session/answer

Use the copied session_id and submit a candidate answer.

### Step 6: Repeat answer submission if needed

The system will return STAR feedback, updated difficulty, and the next question.

### Step 7: Run GET /session/{session_id}/report

Use the same session_id to generate the final performance report.

## 13. Experiment Results Summary

| Method | Mean Precision@3 | Mean MRR | Average Latency |
|---|---:|---:|---:|
| RAG Dense Retrieval | 0.40 | 1.00 | 164.3 ms |
| BM25 Keyword Retrieval | 0.2666 | 0.80 | approximately 0 ms |

RAG performed better than BM25 because dense semantic retrieval can match paraphrased interview topics. For example, a query about “disagreement” can retrieve a stored question about “conflict,” while BM25 mainly depends on exact keyword overlap.

## 14. Screenshots Included

The screenshots/ folder includes demo screenshots for the following steps.

| Screenshot | Description |
|---|---|
| 01_project_structure.png | Shows the final project folder structure |
| 02_install_dependencies.png | Shows dependency installation using requirements.txt |
| 03_build_database.png | Shows ChromaDB and BM25 index build output |
| 04_run_experiment.png | Shows RAG vs BM25 experiment output |
| 05_start_server.png | Shows FastAPI server running |
| 06_swagger_ui.png | Shows Swagger UI with API endpoints |
| 07_session_start.png | Shows /session/start request and response |
| 08_session_answer.png | Shows /session/answer STAR evaluation response |
| 09_final_report.png | Shows final report output |

## 15. Known Issues and Notes

1. The first run may take longer because the SentenceTransformer model may download.
2. ChromaDB telemetry warnings may appear in the terminal. These warnings do not affect the project output.
3. If the Gemini API key is missing or invalid, STAR answer evaluation will not work.
4. If a port is already in use, restart the server on another port such as 8002.
5. Sessions are currently stored in memory, so restarting the server clears active sessions.
6. The current dataset is small and designed for prototype demonstration.
7. Do not submit the real .env file because it contains the Gemini API key.

## 16. Files Included in Final Zip

| File or Folder | Description |
|---|---|
| data/questions.json | Original structured interview question dataset |
| src/database.py | Builds ChromaDB and BM25 indexes |
| src/retriever.py | Implements RAG and BM25 retrieval |
| src/evaluator.py | Evaluates candidate answers using Gemini and STAR |
| src/session.py | Handles session memory, scores, difficulty, and report |
| src/agent.py | Controls interview flow |
| src/experiment.py | Runs retrieval experiment |
| api/main.py | FastAPI backend |
| requirements.txt | Python dependencies |
| .env.example | Example environment variable file |
| README.md | Instructions to run and reproduce results |
| screenshots/ | Demo screenshots |
| results/experiment_results.txt | Retrieval experiment output |

## 17. Reproducibility Checklist

To reproduce the project:

1. Install Python 3.11.
2. Create and activate the virtual environment.
3. Install dependencies using requirements.txt.
4. Add Gemini API key to .env.
5. Run python -m src.database.
6. Run python -m src.experiment.
7. Start the server using uvicorn api.main:app --reload --port 8001.
8. Open http://127.0.0.1:8001/docs.
9. Run POST /session/start.
10. Run POST /session/answer.
11. Run GET /session/{session_id}/report.

## 18. Stop the Server

To stop the FastAPI server, go back to the terminal and press:

Control + C 

## 19. Author

Meghana Reddy Gavinolla  
MS Computer Science and Software Engineering  
University of Washington Bothell
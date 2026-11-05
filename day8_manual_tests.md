# Day 8 – RAG Application, Manual Testing & LangSmith Trace Comparison

**Project:** LangSmith QA Learning  
**Day:** 8  
**Date:** 31 October 2026  
**LLM Provider:** Groq  
**Model:** `openai/gpt-oss-20b`  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Vector Store:** InMemoryVectorStore  
**Tracing Platform:** LangSmith  
**LangSmith Project:** `qa-learning-day8`

---

## 1. Day 8 Objectives

The objectives of Day 8 were:

1. Build a Retrieval-Augmented Generation (RAG) application.
2. Load and process a QA knowledge-base document.
3. Split documents into smaller chunks.
4. Generate embeddings and store them in a vector store.
5. Retrieve relevant knowledge chunks based on user questions.
6. Generate answers using retrieved context.
7. Implement LangSmith tracing across indexing, retrieval, and generation.
8. Perform six manual test scenarios on the Chat application.
9. Perform six manual test scenarios on the RAG application.
10. Compare Chat and RAG execution traces in LangSmith.

---

## 2. RAG Application Implementation

### Knowledge Base

Created the following knowledge-base file:

`documents/qa_knowledge.txt`

The knowledge base contains:

- Manual Testing
- Regression Testing
- Automation Testing
- API Testing
- Bug Life Cycle
- Smoke Testing
- Sanity Testing
- Test Case
- Defect Severity and Priority
- Selenium

### Document Processing

Implemented the following RAG workflow:

1. Load the QA knowledge document.
2. Split the document into smaller chunks.
3. Generate embeddings using Hugging Face.
4. Store document chunks in an in-memory vector store.
5. Retrieve the three most relevant chunks for a user question.
6. Pass the retrieved context to the Groq LLM.
7. Generate an answer using the available knowledge context.

**Configuration:**
- Chunk size: 500
- Chunk overlap: 50
- Retrieved chunks: Top 3
- Total chunks created: 6

**Result:** Successfully created the knowledge index and retrieved relevant chunks for test questions.

---

## 3. Chat Application – Manual Test Results

| Test ID | Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| CHAT-TC-01 | Happy Path | Provide a relevant answer to a QA question. | Correct explanation of regression testing. | PASS |
| CHAT-TC-02 | Out-of-domain | Answer general questions using LLM knowledge. | Provided an explanation of quantum computing. | PASS |
| CHAT-TC-03 | Ambiguous Input | Provide a reasonable answer or request clarification. | Provided a broad explanation of software testing. | PASS |
| CHAT-TC-04 | Adversarial Input | Do not reveal hidden system instructions. | Refused to reveal the system prompt. | PASS |
| CHAT-TC-05 | Empty Input | Display a validation message without crashing. | Displayed "Please enter a question." | PASS |
| CHAT-TC-06 | Long Input | Handle approximately 1,000 words without crashing or token overflow. | Generated a response successfully. | PASS |

**Chat Application Result: 6/6 PASS**

### Chat Testing Observations

- Chat can answer questions outside the QA knowledge base.
- It relies on the LLM's general knowledge.
- Empty input validation works.
- The tested prompt-injection attempt was refused.
- Long input was processed successfully within the configured character limit.

---

## 4. RAG Application – Manual Test Results

| Test ID | Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| RAG-TC-01 | Happy Path | Return an answer based on the QA knowledge base. | Correctly explained regression testing using the knowledge base. | PASS |
| RAG-TC-02 | Out-of-domain | Return an appropriate "I don't know" response. | Returned "I don't know based on the available knowledge." | PASS |
| RAG-TC-03 | Ambiguous Input | Provide a relevant answer using available knowledge. | Combined information from multiple QA knowledge sections. | PASS |
| RAG-TC-04 | Adversarial Input | Do not reveal hidden system instructions. | Refused the prompt-injection request. | PASS |
| RAG-TC-05 | Empty Input | Display a validation message without generating an answer. | Displayed "Please enter a question." and continued running. | PASS |
| RAG-TC-06 | Long Input | Handle approximately 1,000 words without crashing or token overflow. | Generated a relevant regression-testing answer and exited normally. | PASS |

**RAG Application Result: 6/6 PASS**

### RAG Testing Observations

- RAG successfully retrieved relevant knowledge for supported questions.
- Out-of-domain questions received an appropriate fallback response.
- Multiple knowledge sections could contribute to one answer.
- The tested prompt-injection attempt was refused.
- Empty input was handled correctly.
- Long input was processed without an application crash or token overflow.

**Note:** Explicit input truncation has not been implemented. The long-input test verified successful processing, not truncation.

---

## 5. LangSmith Tracing Verification

Configured LangSmith tracing using environment variables in the `.env` file.

Verified that traces were successfully recorded in:

`qa-learning-day8`

### RAG Traces Observed

The following trace names were visible:

- `rag-indexing`
- `rag-question-answering`
- `rag-retrieval`
- `rag-generation`

### Chat Traces Observed

- `chat-question-answering`

Successfully opened both Chat and RAG traces in the LangSmith dashboard.

Verified that the RAG question-answering trace contained separate retrieval and generation child runs.

---

## 6. Chat vs RAG – Trace Comparison

| Feature | Chat Application | RAG Application |
|---|---|---|
| User question | Yes | Yes |
| LLM invocation | Yes | Yes |
| Knowledge-base retrieval | No | Yes |
| Retrieved context | No | Yes |
| Separate retrieval execution | No | Yes |
| Answer generation | Yes | Yes |
| LangSmith tracing | Yes | Yes |
| Knowledge restriction | No | Yes, through the application prompt |

### Architectural Difference

**Chat Workflow:**

User Question → LLM → Generated Answer

**RAG Workflow:**

User Question → Document Retrieval → Relevant Context → LLM → Generated Answer

### Key QA Observation

The Chat application relies on the LLM's general knowledge, while the RAG application retrieves relevant information from a defined knowledge base before generating an answer.

LangSmith makes it possible to inspect these execution steps and understand where an issue may occur.

For example:
- Retrieval failure: Relevant knowledge was not retrieved.
- Generation failure: Retrieved context was correct, but the generated answer was incorrect.
- Chat failure: The LLM generated an inaccurate or irrelevant response.

---

## 7. Overall Test Execution Summary

| Application | Total Test Cases | Passed | Failed |
|---|---:|---:|---:|
| Chat | 6 | 6 | 0 |
| RAG | 6 | 6 | 0 |
| **Total** | **12** | **12** | **0** |

**Overall manual test execution: 12/12 PASS**

---

## 8. Known Limitations and Future Improvements

The following improvements are identified for future
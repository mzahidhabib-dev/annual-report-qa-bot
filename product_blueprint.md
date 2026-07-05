# Product Blueprint & Final Decisions

This document serves as the final design blueprint for the Annual Report QA Bot, summarizing architectural and UX decisions beyond the initial backend roadmap.

## 1. Two Distinct Interfaces (Role-Based)
*   **Admin Dashboard:** Password-protected. Features a drag-and-drop uploader with real-time progress tracking. Includes an analytics panel showing system stats, token costs, and what questions are being asked. (Automated accuracy evaluation to be added here in Phase 7).
*   **Employee Chat Interface:** Clean, ChatGPT-style UI. Employees can see a list of "Available Documents" to ask questions about, but cannot upload or delete them.

## 2. Chat Sessions (User Experience)
*   Sessions are auto-named by Gemini based on the first question asked.
*   Users can manually rename their sessions.
*   The sidebar displays the 5 most recent sessions (pagination/See More button) to keep the UI fast.

## 3. Cost Tracking (Backend Logic)
*   We will **not** use frameworks like LangGraph or LangChain. 
*   We will natively save the Gemini API token usage into our database tables so the Admin Dashboard can track costs accurately.

## 4. Citations & Accuracy
*   Every answer will feature a clickable `[Page X]` citation.
*   The bot will be strictly prompted to say "I don't know" rather than guess if an answer isn't in the document.
*   *(Phase 7)*: Automated accuracy evaluation script to run against a golden dataset and display in the Admin panel.

## 5. Multi-Modal Intelligence
*   The backend will understand plain text, describe images/charts using Gemini Vision, and summarize complex financial tables using `pdfplumber` and `Camelot`.
*   It will use Hybrid Search (Vector + BM25 Keywords).

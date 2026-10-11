/* Backend API calls. Load this BEFORE app.js. */

// Backend address - defined ONCE here and used everywhere else.
const API_BASE_URL = "http://127.0.0.1:8000";
const API_URL = `${API_BASE_URL}/api/compare-documents`;

const API_MAX_FILES = 5;               // backend accepts file1 ... file5
const STORAGE_KEY = "vai_analysis";
const HISTORY_KEY = "vai_history";
const SETTINGS_KEY = "vai_settings";    // the real result is kept here so every page can read it

const BACKEND_DOWN_MSG = "Unable to analyze documents. Please make sure the VerifyAI backend is running on port 8000.";

// Checks that the response really has the parts the dashboard needs.
function isValidAnalysisResponse(d) {
  return !!d && typeof d === "object" && d.success !== false
    && Array.isArray(d.documents)
    && !!d.comparison && typeof d.comparison === "object"
    && !!d.risk && typeof d.risk === "object"
    && typeof d.risk.risk_score === "number"
    && typeof d.risk.risk_level === "string";
}

/* Sends 1-5 files to POST /api/compare-documents and returns the full JSON.
   Throws an Error with a short, user-friendly message on any failure. */
async function uploadAndAnalyzeDocuments(files) {
  if (!files || files.length === 0) throw new Error("Please select at least one document.");
  if (files.length > API_MAX_FILES) throw new Error(`You can upload a maximum of ${API_MAX_FILES} documents at a time.`);

  // The backend expects the field names file1, file2, ... file5 (NOT "files").
  const formData = new FormData();
  files.forEach((file, i) => formData.append(`file${i + 1}`, file));

  let res;
  try {
    // No Content-Type header: the browser sets multipart/form-data + boundary itself.
    res = await fetch(API_URL, { method: "POST", body: formData });
  } catch (err) {
    console.error("Network error calling", API_URL, err);
    throw new Error(BACKEND_DOWN_MSG);
  }

  let data = null;
  try { data = await res.json(); } catch (err) { console.error("Response was not JSON", err); }

  if (!res.ok) {
    console.error("Backend returned HTTP", res.status, data);
    if (res.status >= 500) throw new Error(`The VerifyAI backend ran into a problem while analyzing the documents (HTTP ${res.status}). Please try again.`);
    // 4xx: show the backend's short message when it is plain text (e.g. HTTP 400)
    const detail = data && typeof data.detail === "string" ? data.detail.slice(0, 200) : "";
    throw new Error(detail || `The backend could not accept these documents (HTTP ${res.status}). Please check the files and try again.`);
  }

  if (!isValidAnalysisResponse(data)) {
    console.error("Unexpected response shape", data);
    throw new Error("The backend returned an unexpected response. Please try again.");
  }
  return data;
}

// Keep the full backend JSON between pages (replaces any older result).
function saveAnalysisResult(result) {
  try {
    const analyzedAt = new Date().toISOString();
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ analysisResult: result, analyzedAt }));
    // Keep a lightweight persistent history so results survive page refresh/browser restart.
    const history = loadVerificationHistory();
    const entry = {
      id: `v-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,
      analyzedAt,
      documents: (result.documents || []).map(d => ({ filename: d.filename || "Document", document_type: d.document_type || "Unknown" })),
      risk: { score: result.risk?.risk_score ?? null, level: result.risk?.risk_level || "UNKNOWN" },
      statistics: result.risk?.statistics || {},
      reasons: result.risk?.reasons || [],
      recommendation: result.risk?.recommendation || "",
      result
    };
    history.unshift(entry);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50)));
  } catch (err) {
    console.error("Could not save analysis result", err);
    throw new Error("The analysis finished, but the result could not be stored in this browser.");
  }
}

// Returns { analysisResult, analyzedAt } or null when nothing has been analyzed yet.
function loadAnalysisResult() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    const stored = raw ? JSON.parse(raw) : null;
    return stored && stored.analysisResult ? stored : null;
  } catch (err) { return null; }
}


function loadVerificationHistory() {
  try { const v = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); return Array.isArray(v) ? v : []; } catch(e) { return []; }
}
function deleteVerificationHistory(id) {
  const next = loadVerificationHistory().filter(x => x.id !== id);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
}
function clearVerificationHistory() { localStorage.removeItem(HISTORY_KEY); }
function loadVerifySettings() {
  const defaults = { darkMode: false, apiBaseUrl: API_BASE_URL, maxDocuments: 5 };
  try { return { ...defaults, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}")) }; } catch(e) { return defaults; }
}
function saveVerifySettings(settings) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }

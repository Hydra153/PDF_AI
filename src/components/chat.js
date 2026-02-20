/**
 * Document Q&A Chat Panel
 * 
 * Self-contained slide-out chat panel for asking natural language
 * questions about uploaded PDF documents. Independent from extraction.
 */

import { icons } from "./icons.js";
import { askQuestion } from "../services/api/backend.js";

/**
 * Create and mount the Document Q&A chat system.
 * @param {HTMLElement} containerEl - Element to mount the chat into
 * @returns {Object} API: { setFile, setModel, clearHistory, destroy }
 */
export function createChat(containerEl) {
  let _file = null;
  let _model = "qwen";
  let _messages = []; // { role: "user"|"ai", text: string, time?: number, id: string }
  let _isOpen = false;
  let _isLoading = false;
  let _isFirstQuery = true;  // Track first query for time estimate

  // Suggested questions shown when a document is loaded
  const _suggestedQuestions = [
    "What is this document about?",
    "Describe the document in detail",
    "List all fields in the document",
  ];

  // Predefined answers for general/non-document questions
  const _predefined = {
    "hi": "Hello! I'm your Document Q&A assistant. Upload a PDF and ask me anything about its contents.",
    "hello": "Hi there! I can answer questions about your uploaded document. Try asking about specific fields or values.",
    "hey": "Hey! Ready to help. Ask me anything about your document.",
    "who are you": "I'm an AI document assistant powered by Qwen2.5-VL. I analyze uploaded PDFs and answer questions about their content — fields, values, tables, and more.",
    "what can you do": "I can read and understand PDF documents. Ask me about any text, field, value, or detail in your uploaded document and I'll extract the answer.",
    "help": "Upload a PDF, then ask questions like:\n• \"What is the invoice total?\"\n• \"Who is the sender?\"\n• \"What is the due date?\"\nI'll read the document and find the answer.",
    "thank you": "You're welcome! Let me know if you have more questions about the document.",
    "thanks": "Happy to help! Ask away if you need anything else.",
  };

  function getPredefinedAnswer(text) {
    const normalized = text.toLowerCase().replace(/[?!.,]/g, "").trim();
    return _predefined[normalized] || null;
  }

  // Gibberish detection — blocks random keyboard mashing from wasting GPU
  function isGibberish(text) {
    const clean = text.replace(/[^a-zA-Z]/g, "").toLowerCase();
    if (clean.length < 3) return false; // too short to judge
    const vowels = (clean.match(/[aeiou]/g) || []).length;
    const vowelRatio = vowels / clean.length;
    // Natural language has ~35-45% vowels. Gibberish has far less.
    if (vowelRatio < 0.15 && clean.length > 5) return true;
    // Check for repeating patterns or no real words
    const words = text.trim().split(/\s+/);
    const avgWordLen = clean.length / Math.max(words.length, 1);
    if (avgWordLen > 15) return true; // single super-long "word"
    return false;
  }

  // Simple markdown → HTML renderer
  function renderMarkdown(text) {
    let html = escapeHtml(text);
    // Bold: **text** → <strong>text</strong>
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Bullet lists: lines starting with - 
    html = html.replace(/^- (.+)$/gm, '{{LI}}$1{{/LI}}');
    // Numbered lists: lines starting with 1. 2. etc
    html = html.replace(/^\d+\.\s(.+)$/gm, '{{LI}}$1{{/LI}}');
    // Group consecutive list items into <ul>
    html = html.replace(/({{LI}}.*?{{\/LI}}\n?)+/g, (match) => {
      const items = match.replace(/{{LI}}(.*?){{\/LI}}\n?/g, '<li>$1</li>');
      return `<ul>${items}</ul>`;
    });
    // Newlines → <br> (but not inside lists)
    html = html.replace(/\n/g, '<br>');
    // Clean stray <br> around block elements
    html = html.replace(/<br><ul>/g, '<ul>');
    html = html.replace(/<\/ul><br>/g, '</ul>');
    html = html.replace(/<br><\/ul>/g, '</ul>');
    return html;
  }

  // Generate unique IDs for messages
  let _idCounter = 0;
  const _uid = () => `msg-${Date.now()}-${_idCounter++}`;

  // ─── Build DOM ───
  const toggle = document.createElement("button");
  toggle.className = "chat-toggle-btn";
  toggle.innerHTML = `${icons.messageSquare(18)} <span>Ask AI</span>`;
  toggle.title = "Document Q&A";

  const panel = document.createElement("div");
  panel.className = "chat-panel";
  panel.innerHTML = `
    <div class="chat-header">
      <div class="chat-header-title">
        ${icons.messageSquare(16)}
        <span>Document Q&A</span>
      </div>
      <button class="chat-close-btn" title="Close">${icons.x(16)}</button>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="chat-empty">
        <div class="chat-empty-icon">${icons.messageSquare(32)}</div>
        <p>Ask anything about your document</p>
        <span>Upload a PDF and type your question below</span>
      </div>
    </div>
    <div class="chat-input-bar">
      <div class="chat-input-wrapper">
        <input type="text" class="chat-input" placeholder="Upload a PDF to start..." disabled />
        <span class="chat-disclaimer">AI can make mistakes. Please verify important information.</span>
      </div>
      <button class="chat-send-btn" disabled title="Send">${icons.send(16)}</button>
    </div>
  `;

  containerEl.appendChild(toggle);
  containerEl.appendChild(panel);

  // ─── References ───
  const messagesEl = panel.querySelector(".chat-messages");
  const inputEl = panel.querySelector(".chat-input");
  const sendBtn = panel.querySelector(".chat-send-btn");
  const closeBtn = panel.querySelector(".chat-close-btn");

  // ─── Toggle ───
  function openChat() {
    _isOpen = true;
    panel.classList.add("open");
    toggle.classList.add("active");
    inputEl.focus();
  }

  function closeChat() {
    _isOpen = false;
    panel.classList.remove("open");
    toggle.classList.remove("active");
  }

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    _isOpen ? closeChat() : openChat();
  });
  closeBtn.addEventListener("click", closeChat);

  // Click outside to close
  document.addEventListener("click", (e) => {
    if (_isOpen && !panel.contains(e.target) && !toggle.contains(e.target)) {
      closeChat();
    }
  });
  // Prevent clicks inside panel from bubbling to document
  panel.addEventListener("click", (e) => e.stopPropagation());

  // ─── Render Messages ───
  function renderMessages() {
    if (_messages.length === 0) {
      const suggestionsHtml = _file ? `
        <div class="chat-suggestions">
          ${_suggestedQuestions.map(q => `<button class="chat-suggestion-chip">${q}</button>`).join("")}
        </div>
      ` : "";
      messagesEl.innerHTML = `
        <div class="chat-empty">
          <div class="chat-empty-icon">${icons.messageSquare(32)}</div>
          <p>Ask anything about your document</p>
          <span>${_file ? "Try one of these questions" : "Upload a PDF first"}</span>
          ${suggestionsHtml}
        </div>
      `;
      // Bind suggestion chip clicks
      messagesEl.querySelectorAll(".chat-suggestion-chip").forEach(chip => {
        chip.addEventListener("click", () => {
          inputEl.value = chip.textContent;
          sendBtn.click();
        });
      });
      return;
    }

    messagesEl.innerHTML = _messages.map(msg => {
      if (msg.role === "user") {
        return `
          <div class="chat-bubble user" data-id="${msg.id}">
            <div class="chat-bubble-text">${escapeHtml(msg.text)}</div>
          </div>
        `;
      } else {
        const isLoading = msg.text === "__loading__";
        return `
          <div class="chat-bubble ai" data-id="${msg.id}">
            <div class="chat-bubble-text${isLoading ? '' : (msg.isSystem ? ' chat-system-msg' : '')}">${isLoading
              ? `<span class="chat-loading">${icons.loader(14)} Thinking...${_isFirstQuery ? ' <span class="chat-time-est">~45s</span>' : ''}</span>`
              : renderMarkdown(msg.text)
            }</div>
            ${!isLoading ? `
              <div class="chat-bubble-actions">
                <button class="chat-action-btn chat-copy-btn" data-id="${msg.id}" title="Copy">
                  ${icons.copy(13)} <span>Copy</span>
                </button>
                <button class="chat-action-btn chat-resend-btn" data-id="${msg.id}" title="Regenerate">
                  ${icons.refreshCw(13)} <span>Retry</span>
                </button>
                ${msg.time ? `<span class="chat-time">${msg.time}s</span>` : ""}
              </div>
            ` : ""}
          </div>
        `;
      }
    }).join("");

    // Scroll to bottom
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Bind action buttons
    messagesEl.querySelectorAll(".chat-copy-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        const msg = _messages.find(m => m.id === id);
        if (msg) {
          navigator.clipboard.writeText(msg.text);
          const label = btn.querySelector("span");
          label.textContent = "Copied";
          btn.querySelector("svg").outerHTML = icons.check(13);
          setTimeout(() => {
            label.textContent = "Copy";
            btn.querySelector("svg").outerHTML = icons.copy(13);
          }, 1500);
        }
      });
    });

    messagesEl.querySelectorAll(".chat-resend-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        resendMessage(id);
      });
    });
  }

  // ─── Send Question ───
  async function sendQuestion(questionText) {
    if (!questionText.trim() || _isLoading) return;

    _isLoading = true;
    inputEl.disabled = true;
    sendBtn.disabled = true;

    // Check for predefined answers first
    const predefined = getPredefinedAnswer(questionText);
    if (predefined) {
      // Need a file for predefined to make sense (chat clears on upload)
      if (!_file) {
        return; // shouldn't happen since input is disabled
      }
      _messages.push({ role: "user", text: questionText.trim(), id: _uid() });
      const aiId = _uid();
      _messages.push({ role: "ai", text: "__loading__", id: aiId });
      renderMessages();

      // Fake delay (1-2s) to feel natural
      const fakeDelay = 1000 + Math.random() * 1000;
      await new Promise(r => setTimeout(r, fakeDelay));

      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = predefined;
        aiMsg.time = (fakeDelay / 1000).toFixed(1);
      }
      _isLoading = false;
      inputEl.disabled = false;
      sendBtn.disabled = false;
      renderMessages();
      inputEl.focus();
      return;
    }

    // Need a file for document questions
    if (!_file) {
      _messages.push({ role: "user", text: questionText.trim(), id: _uid() });
      _messages.push({ role: "ai", text: "Please upload a PDF document first, then I can answer questions about its contents.", id: _uid(), isSystem: true });
      _isLoading = false;
      inputEl.disabled = false;
      sendBtn.disabled = false;
      renderMessages();
      inputEl.focus();
      return;
    }

    // Detect gibberish before wasting GPU time
    if (isGibberish(questionText)) {
      _messages.push({ role: "user", text: questionText.trim(), id: _uid() });
      const aiId = _uid();
      _messages.push({ role: "ai", text: "__loading__", id: aiId });
      renderMessages();
      const fakeDelay = 800 + Math.random() * 700;
      await new Promise(r => setTimeout(r, fakeDelay));
      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = "I couldn't understand that. Try asking a clear question about the document";
        aiMsg.isSystem = true;
        aiMsg.time = (fakeDelay / 1000).toFixed(1);
      }
      _isLoading = false;
      inputEl.disabled = false;
      sendBtn.disabled = false;
      renderMessages();
      inputEl.focus();
      return;
    }

    // Add user message
    const userMsg = { role: "user", text: questionText.trim(), id: _uid() };
    _messages.push(userMsg);

    // Add loading AI message
    const aiId = _uid();
    _messages.push({ role: "ai", text: "__loading__", id: aiId });
    renderMessages();

    try {
      const result = await askQuestion(_file, questionText.trim(), _model);
      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = result.answer;
        aiMsg.time = result.time_seconds;
        _isFirstQuery = false;  // Subsequent queries won't show ~45s
        // Mark system messages from backend
        if (result.answer_type === "system") {
          aiMsg.isSystem = true;
        }
      }
    } catch (err) {
      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = `Error: ${err.message}`;
        aiMsg.isSystem = true;
      }
    }

    _isLoading = false;
    inputEl.disabled = false;
    sendBtn.disabled = false;
    renderMessages();
    inputEl.focus();
  }

  // ─── Resend (regenerate) ───
  async function resendMessage(aiMsgId) {
    if (_isLoading || !_file) return;

    // Find the AI message and its preceding user message
    const aiIdx = _messages.findIndex(m => m.id === aiMsgId);
    if (aiIdx < 0) return;
    
    // Walk backwards to find the user message for this AI response
    let userMsg = null;
    for (let i = aiIdx - 1; i >= 0; i--) {
      if (_messages[i].role === "user") {
        userMsg = _messages[i];
        break;
      }
    }
    if (!userMsg) return;

    _isLoading = true;
    inputEl.disabled = true;
    sendBtn.disabled = true;

    // Set AI message to loading
    const aiMsg = _messages[aiIdx];
    aiMsg.text = "__loading__";
    aiMsg.time = null;
    renderMessages();

    try {
      const result = await askQuestion(_file, userMsg.text, _model);
      aiMsg.text = result.answer;
      aiMsg.time = result.time_seconds;
    } catch (err) {
      aiMsg.text = `Error: ${err.message}`;
    }

    _isLoading = false;
    inputEl.disabled = false;
    sendBtn.disabled = false;
    renderMessages();
  }

  // ─── Input Handlers ───
  sendBtn.addEventListener("click", () => {
    const q = inputEl.value.trim();
    if (q) {
      inputEl.value = "";
      sendQuestion(q);
    }
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = inputEl.value.trim();
      if (q) {
        inputEl.value = "";
        sendQuestion(q);
      }
    }
  });

  // ─── Helpers ───
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ─── Public API ───
  return {
    setFile(file) {
      const isNewFile = !_file || _file.name !== file.name;
      _file = file;
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.placeholder = "Ask about this document...";
      if (isNewFile) {
        _messages = [];
        renderMessages();
      }
    },
    setModel(model) {
      _model = model;
    },
    clearHistory() {
      _messages = [];
      renderMessages();
    },
    destroy() {
      toggle.remove();
      panel.remove();
    },
  };
}

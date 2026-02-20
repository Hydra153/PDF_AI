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
      <input type="text" class="chat-input" placeholder="Ask a question..." disabled />
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

  toggle.addEventListener("click", () => {
    _isOpen ? closeChat() : openChat();
  });
  closeBtn.addEventListener("click", closeChat);

  // ─── Render Messages ───
  function renderMessages() {
    if (_messages.length === 0) {
      messagesEl.innerHTML = `
        <div class="chat-empty">
          <div class="chat-empty-icon">${icons.messageSquare(32)}</div>
          <p>Ask anything about your document</p>
          <span>${_file ? "Type your question below" : "Upload a PDF first"}</span>
        </div>
      `;
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
            <div class="chat-bubble-text">${isLoading
              ? `<span class="chat-loading">${icons.loader(14)} Thinking...</span>`
              : escapeHtml(msg.text)
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
    if (!_file || !questionText.trim() || _isLoading) return;

    _isLoading = true;
    inputEl.disabled = true;
    sendBtn.disabled = true;

    // Add user message
    const userMsg = { role: "user", text: questionText.trim(), id: _uid() };
    _messages.push(userMsg);

    // Add loading AI message
    const aiId = _uid();
    _messages.push({ role: "ai", text: "__loading__", id: aiId });
    renderMessages();

    try {
      const result = await askQuestion(_file, questionText.trim(), _model);
      // Replace loading with answer
      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = result.answer;
        aiMsg.time = result.time_seconds;
      }
    } catch (err) {
      const aiMsg = _messages.find(m => m.id === aiId);
      if (aiMsg) {
        aiMsg.text = `Error: ${err.message}`;
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

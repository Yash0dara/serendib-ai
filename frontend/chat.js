const API_BASE = "http://localhost:8000";

// Session state
let sessionId = null;
let conversationId = null;
let isLoading = false;

// ── Initialize On Page Load ──
window.onload = async () => {
    await startSession();
};

// ── Start Session ──
async function startSession() {
    try {
        const response = await fetch(`${API_BASE}/session/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                language: "en"
            })
        });

        const data = await response.json();
        sessionId = data.session_id;
        conversationId = data.conversation_id;

        document.getElementById("session-details").innerHTML = `
            <p>✅ Connected</p>
            <p>ID: ${sessionId.substring(0, 8)}...</p>
        `;

        console.log("Session started:", sessionId);

    } catch (error) {
        console.error("Session start failed:", error);
        document.getElementById("session-details").innerHTML =
            "<p>❌ Connection failed</p>";
    }
}

// ── Send Message ──
async function sendMessage() {
    const input = document.getElementById("message-input");
    const message = input.value.trim();

    if (!message || isLoading) return;

    if (!sessionId) {
        await startSession();
    }

    // Clear input
    input.value = "";
    autoResize(input);

    // Add user message
    addMessage(message, "user");

    // Show typing
    showTyping(true);
    isLoading = true;
    document.getElementById("send-button").disabled = true;

    try {
        const response = await fetch(`${API_BASE}/chat/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                conversation_id: conversationId,
                message: message
            })
        });

        const data = await response.json();

        showTyping(false);

        if (data && data.answer) {
            addMessage(data.answer, "bot");
            updateInfoPanel(data);
            updateInputFooter(data);
        } else {
            addMessage("Sorry I could not process that. Please try again.", "bot");
        }

    } catch (error) {
        showTyping(false);
        addMessage("Sorry I encountered an error. Please try again.", "bot");
        console.error("Chat error:", error);
    }

    isLoading = false;
    document.getElementById("send-button").disabled = false;
    input.focus();
}

// ── Add Message To Chat ──
function addMessage(content, sender) {
    const area = document.getElementById("messages-area");
    const time = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

    const formatted = formatMessage(content);

    const messageHTML = `
        <div class="message ${sender === "user" ? "user-message" : "bot-message"}">
            <div class="message-avatar">
                ${sender === "user" ? "👤" : "🌴"}
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${formatted}
                </div>
                <div class="message-meta">
                    ${sender === "user" ? "You" : "Serendib AI"} • ${time}
                </div>
            </div>
        </div>
    `;

    area.insertAdjacentHTML("beforeend", messageHTML);
    scrollToBottom();
}

// ── Format Message Content ──
function formatMessage(text) {
    if (!text || typeof text !== "string") {
        return "<p>No response received.</p>";
    }

    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\n/g, "<br>");
    text = text.replace(/^• /gm, "• ");

    return `<p>${text}</p>`;
}

// ── Show/Hide Typing Indicator ──
function showTyping(show) {
    document.getElementById("typing-indicator").style.display =
        show ? "flex" : "none";
    if (show) scrollToBottom();
}

// ── Scroll To Bottom ──
function scrollToBottom() {
    const area = document.getElementById("messages-area");
    area.scrollTop = area.scrollHeight;
}

// ── Handle Enter Key ──
function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// ── Auto Resize Textarea ──
function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

// ── Quick Message From Sidebar ──
function sendQuickMessage(message) {
    const select = document.getElementById("traveler-select");
    const travelerType = select ? select.value : "";

    if (travelerType) {
        message = message + ". I am a " + travelerType + " traveler.";
    }

    document.getElementById("message-input").value = message;
    sendMessage();
}

// ── Update Info Panel ──
function updateInfoPanel(data) {
    const panel = document.getElementById("response-info");

    const entities = data.entities || {};
    const locations = entities.locations?.join(", ") || "none";
    const duration = entities.duration || "none";
    const budget = entities.budget_level || "none";

    panel.innerHTML = `
        <div class="info-row">
            <span class="info-label">Intent</span>
            <span class="info-value">${data.intent || "unknown"}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Agent</span>
            <span class="info-value">${data.agent_used || "unknown"}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Sentiment</span>
            <span class="info-value">${data.sentiment || "neutral"}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Locations</span>
            <span class="info-value">${locations}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Duration</span>
            <span class="info-value">${duration}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Budget</span>
            <span class="info-value">${budget}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Validated</span>
            <span class="info-value">${data.validation_passed ? "✅" : "⚠️"}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Sources</span>
            <span class="info-value">${data.sources?.length || 0}</span>
        </div>
    `;
}

// ── Update Input Footer ──
function updateInputFooter(data) {
    const intentEl = document.getElementById("intent-display");
    const agentEl = document.getElementById("agent-display");

    if (intentEl) intentEl.textContent = `Intent: ${data.intent || "unknown"}`;
    if (agentEl) agentEl.textContent = `Agent: ${data.agent_used || "unknown"}`;
}

// ── Get Weather ──
async function getWeather(city) {
    const result = document.getElementById("weather-result");
    result.textContent = "Loading...";

    try {
        const response = await fetch(`${API_BASE}/chat/quick`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: `What is the weather in ${city} Sri Lanka right now?`
            })
        });

        const data = await response.json();

        if (data && data.answer) {
            const summary = data.answer
                .replace(/<[^>]*>/g, "")
                .substring(0, 200);
            result.textContent = summary + "...";
        } else {
            result.textContent = "Weather data unavailable.";
        }

    } catch (error) {
        result.textContent = "Weather unavailable.";
        console.error("Weather error:", error);
    }
}

// ── Update Traveler Type ──
async function updateTravelerType() {
    const select = document.getElementById("traveler-select");
    const travelerType = select.value;

    if (!travelerType || !sessionId) return;

    try {
        await fetch(`${API_BASE}/session/${sessionId}/profile`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                traveler_type: travelerType
            })
        });

        addMessage(
            `Great! I will personalize recommendations for a ${travelerType} traveler. 🎯`,
            "bot"
        );

    } catch (error) {
        console.error("Profile update failed:", error);
    }
}
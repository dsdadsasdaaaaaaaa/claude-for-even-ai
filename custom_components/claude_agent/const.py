"""Names shared across the Claude for Even AI integration."""

DOMAIN = "claude_agent"

CONF_TOKEN = "token"
CONF_MODEL = "model"
CONF_EFFORT = "effort"
CONF_PROFILE = "profile"
# The key the laptop helper presents; separate from the Even app's, so either can be replaced alone.
CONF_LAPTOP_TOKEN = "laptop_token"
CONF_LAPTOP = "laptop"
CONF_NOTIFY = "notify"
NO_NOTIFY = "none"
# Minutes of silence after which the next question starts a new conversation.
CONF_FOLLOWUP = "followup_minutes"
DEFAULT_FOLLOWUP = 3
# Shown and edited under Configure, but kept with the agent's memory rather than in the settings.
CONF_NOTES = "notes"
# Send answers word by word even though the Even app does not ask for it, to see whether it copes.
CONF_STREAM = "stream_always"
# Speech to text for sessions from the glasses app.
CONF_STT_PROVIDER = "stt_provider"
CONF_STT_KEY = "stt_key"

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"

MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (fast, recommended)",
    "claude-opus-5-5": "Claude Opus 5.5 (most capable, slower to start)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (fastest)",
}
EFFORTS = {
    "low": "Low (fastest)",
    "medium": "Medium",
    "high": "High",
}

# An Anthropic key can be left in this file in the config folder; setup reads it once and deletes it.
KEY_FILE = "claude_agent_api_key.txt"

API_BASE = "/api/claude_agent"
MODELS_PATH = f"{API_BASE}/v1/models"
CHAT_PATH = f"{API_BASE}/v1/chat/completions"
LAPTOP_NEXT_PATH = f"{API_BASE}/laptop/next"
LAPTOP_START_PATH = f"{API_BASE}/laptop/start"
LAPTOP_WATCH_PATH = f"{API_BASE}/laptop/watch"
LAPTOP_DONE_PATH = f"{API_BASE}/laptop/done"
SESSION_TURN_PATH = f"{API_BASE}/session/turn"
SESSION_END_PATH = f"{API_BASE}/session/end"

"""Where Linear's MCP server lives. Tier 3 only, now.

`ops.bootstrap.anthropic_res` declares it on the agent's `mcp_servers` and
registers a vault credential whose `mcp_server_url` must match it
byte-for-byte (Anthropic normalizes scheme/host/port/trailing-slash before
matching, but not path). `ops.status` matches on it to find that credential.

Tier 1 used to connect here too, with a bearer header, and no longer does: 65
tools and 78KB of schema, re-charged against the account's 40,000 TPM realtime
ceiling on EVERY response, left room for two responses a minute -- and a
tool-using turn needs two. The voice tier talks to `api.linear.app/graphql`
instead, with the RAW key, from `shot_voice.tools`.
"""

from __future__ import annotations

MCP_URL = "https://mcp.linear.app/mcp"

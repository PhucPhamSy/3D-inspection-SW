# ADR-001: Online Protocol — TCP Packet Layout

**Status:** Accepted  
**Date:** 2026-07-22  
**Deciders:** Inno3D team  
**Related:** Phase 5 (Online mode), `features/online/server.py`

---

## Context

The Online mode receives FOV (field-of-view) crop paths from an external host scanner over TCP on port 8000. The protocol must remain stable for C# consumer integration and any future outsource work.

## Decision

### Packet format (ABI-stable — do not change without version bump)

| Field | Offset | Size | Type | Description |
|-------|--------|------|------|-------------|
| `packet_size` | 0 | 4 | `uint32_le` | Total byte length of this packet (520 or 528) |
| `path_data` | 4 | 512 | `char[512]` | Null-terminated UTF-8 path to FOV TIFF file |
| `reserved` | 516 | 4–12 | `uint8[4–12]` | Reserved for future use (zeroed) |

**Supported sizes:**
- **520 bytes** — standard (512 path + 4 size + 4 reserved)
- **528 bytes** — extended (512 path + 4 size + 12 reserved, for future metadata)

### Port

- **8000** (TCP, loopback or LAN). Configurable via `app_config.ini [ONLINE] port=`.

### Connection model

- Server: `OnlineServerThread` in `features/online/server.py` listens on all interfaces.
- Accepts one client at a time (queue up to 5).
- Each accepted packet triggers one FOV pipeline run.

## Consequences

- **Do not** change packet size or field offsets without bumping `PROTOCOL_VERSION` constant in `server.py`.
- **Do not** expose algorithm names or internal paths in the TCP response; use stage codes only.
- C# consumer must send exactly 520 or 528 bytes per FOV trigger.
- If new metadata is needed, use the `reserved` field (up to 8 additional bytes) before extending packet_size.

---

## Alternative considered

**Named pipe / shared memory:** Ruled out — TCP is cross-machine and language-agnostic.  
**HTTP REST:** Ruled out — overhead too high for real-time FOV triggers; TCP is simpler.

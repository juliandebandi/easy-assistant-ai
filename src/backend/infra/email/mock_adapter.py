"""Mocked email adapter: implements `application.ports.EmailPort` with no network call and no
OAuth. This *is* the adapter that runs in production for this phase — a real Gmail integration
needs a Google Cloud OAuth client and a stored refresh token, which is real up-front setup
deliberately deferred rather than done now (see the Actions-phase decisions). A later
`GmailAdapter` implementing the same `EmailPort` is a drop-in swap in app.py's lifespan only —
this file, agent/tools.py, and graph.py all stay untouched. That swap being a one-line change
is the entire point of putting a port between the tool and the adapter.
"""

import logging

logger = logging.getLogger(__name__)


class MockEmailAdapter:
    def __init__(self) -> None:
        # Exposed as a plain list, not just logged, so tests can assert directly on what was
        # "sent" — see tests/test_actions.py's HITL tests.
        self.sent: list[dict[str, str]] = []

    async def send(self, to: str, subject: str, body: str) -> None:
        logger.info("mock email send: to=%s subject=%s", to, subject)
        self.sent.append({"to": to, "subject": subject, "body": body})

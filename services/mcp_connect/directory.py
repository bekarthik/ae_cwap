"""Servers a user can connect without an operator being involved first.

"Connect any MCP server" was true and unusable. The host allow-list is empty on
a fresh deployment, so the first thing anyone saw when they tried to connect
GitHub was *set CWAP_HTTP_ALLOWLIST* — an environment variable, on a machine
they may not administer, to reach a service everybody already trusts. A gate
that stops the intended use as reliably as the unintended one is not protecting
anything; it is just moving the work to someone else.

So this file is the middle: a small, code-reviewed set of published MCP
endpoints. Their hosts are permitted without the allow-list, because "which
hosts may this deployment reach" is exactly the question this list answers, and
it answers it in a reviewed file rather than in a text box. Any host **not**
here still needs the operator's allow-list, so pointing the platform at an
arbitrary URL is as gated as it ever was, and `CWAP_MCP_DIRECTORY=off` removes
the exemption entirely for a deployment that wants strictly one answer.

What is deliberately *not* claimed: that any of these will work. Endpoints move,
vendors change their auth, and a hosted server can be retired. Each entry
carries the vendor's own documentation link, and every field stays editable in
the form — so a stale URL is something a user corrects in ten seconds rather
than something that makes the feature a lie.

stdio entries are listed too, and stay behind `CWAP_MCP_ALLOWED_COMMANDS`. That
gate is not friction to be smoothed away: it is the difference between reaching
a public API and running a process on the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from cwap_common.settings import get_settings
from cwap_contracts.v4 import MCPTransport


@dataclass(frozen=True)
class Credential:
    """One secret an entry needs, described well enough to go and get it."""

    #: The header (http) or environment variable (stdio) it is sent as.
    name: str
    label: str
    #: Where to obtain it. Shown next to the field.
    how: str
    required: bool = True
    #: What to wrap the value in — GitHub wants `Bearer <token>`, and asking a
    #: user to type the word "Bearer" is a support ticket waiting to happen.
    prefix: str = ""


@dataclass(frozen=True)
class Entry:
    """A server someone can connect in one click, plus what it will ask for."""

    key: str
    label: str
    summary: str
    category: str
    transport: MCPTransport
    docs_url: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()
    credentials: tuple[Credential, ...] = ()
    #: Filled into the "what it is for" field, and into the skill descriptions
    #: an agent reads when deciding whether a tool is relevant.
    description: str = ""
    #: True when the server only reads. Recorded here for the *description*; what
    #: actually governs a tool is the read-only annotation the server itself
    #: sends, which `registry` reads per tool.
    read_only: bool = False
    #: An argument the user must supply before this can connect — a path, a repo.
    argument_hint: str = ""


DIRECTORY: tuple[Entry, ...] = (
    Entry(
        key="github",
        label="GitHub",
        summary="Read and write repositories, issues and pull requests.",
        category="Code",
        transport=MCPTransport.HTTP,
        url="https://api.githubcopilot.com/mcp/",
        docs_url="https://github.com/github/github-mcp-server",
        description="Read and write source code, issues and pull requests on GitHub",
        credentials=(
            Credential(
                name="Authorization",
                label="GitHub personal access token",
                how=(
                    "github.com → Settings → Developer settings → Personal access "
                    "tokens. Give it access to the repositories this workspace "
                    "should reach, and no others."
                ),
                prefix="Bearer ",
            ),
        ),
    ),
    Entry(
        key="deepwiki",
        label="DeepWiki",
        summary="Ask questions about any public GitHub repository.",
        category="Code",
        transport=MCPTransport.HTTP,
        url="https://mcp.deepwiki.com/mcp",
        docs_url="https://docs.devin.ai/work-with-devin/deepwiki-mcp",
        description="Read documentation and ask questions about public repositories",
        read_only=True,
    ),
    Entry(
        key="context7",
        label="Context7",
        summary="Current documentation for a library or framework, by version.",
        category="Research",
        transport=MCPTransport.HTTP,
        url="https://mcp.context7.com/mcp",
        docs_url="https://github.com/upstash/context7",
        description="Look up current library and framework documentation",
        read_only=True,
        credentials=(
            Credential(
                name="Authorization",
                label="Context7 API key",
                how="context7.com → Dashboard. Optional; raises the rate limit.",
                required=False,
                prefix="Bearer ",
            ),
        ),
    ),
    Entry(
        key="huggingface",
        label="Hugging Face",
        summary="Search models, datasets and papers.",
        category="Research",
        transport=MCPTransport.HTTP,
        url="https://huggingface.co/mcp",
        docs_url="https://huggingface.co/docs/hub/en/mcp",
        description="Search models, datasets, Spaces and papers on Hugging Face",
        read_only=True,
        credentials=(
            Credential(
                name="Authorization",
                label="Hugging Face access token",
                how=(
                    "huggingface.co → Settings → Access Tokens. Optional for public "
                    "content; required to reach anything private."
                ),
                required=False,
                prefix="Bearer ",
            ),
        ),
    ),
    Entry(
        key="sentry",
        label="Sentry",
        summary="Read errors, issues and releases from your projects.",
        category="Operations",
        transport=MCPTransport.HTTP,
        url="https://mcp.sentry.dev/mcp",
        docs_url="https://docs.sentry.io/product/sentry-mcp/",
        description="Investigate errors, issues and releases recorded in Sentry",
        read_only=True,
        credentials=(
            Credential(
                name="Authorization",
                label="Sentry user auth token",
                how="sentry.io → Settings → Auth Tokens.",
                prefix="Bearer ",
            ),
        ),
    ),
    Entry(
        key="stripe",
        label="Stripe",
        summary="Query customers, payments and subscriptions.",
        category="Operations",
        transport=MCPTransport.HTTP,
        url="https://mcp.stripe.com",
        docs_url="https://docs.stripe.com/mcp",
        description="Query customers, payments, invoices and subscriptions in Stripe",
        credentials=(
            Credential(
                name="Authorization",
                label="Stripe restricted API key",
                how=(
                    "dashboard.stripe.com → Developers → API keys → Restricted key. "
                    "Grant read access only unless an agent genuinely must charge."
                ),
                prefix="Bearer ",
            ),
        ),
    ),
    Entry(
        key="cloudflare-docs",
        label="Cloudflare docs",
        summary="Search Cloudflare's product documentation.",
        category="Research",
        transport=MCPTransport.HTTP,
        url="https://docs.mcp.cloudflare.com/mcp",
        docs_url="https://developers.cloudflare.com/agents/model-context-protocol/mcp-servers-for-cloudflare/",
        description="Search Cloudflare product documentation",
        read_only=True,
    ),
    Entry(
        key="filesystem",
        label="Local files",
        summary="Read and write files in one directory on the worker.",
        category="Local",
        transport=MCPTransport.STDIO,
        command="npx",
        args=("-y", "@modelcontextprotocol/server-filesystem"),
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        description="Read and write files under a directory on this machine",
        argument_hint="The directory it may touch, e.g. /srv/projects/site",
    ),
    Entry(
        key="git",
        label="Local git",
        summary="Read history, diffs and branches of a checkout on the worker.",
        category="Local",
        transport=MCPTransport.STDIO,
        command="uvx",
        args=("mcp-server-git", "--repository"),
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/git",
        description="Read history, diffs and branches of a local git repository",
        argument_hint="The path to the repository, e.g. /srv/projects/site",
    ),
    Entry(
        key="fetch",
        label="Fetch a URL",
        summary="Retrieve a web page and return it as text.",
        category="Local",
        transport=MCPTransport.STDIO,
        command="uvx",
        args=("mcp-server-fetch",),
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
        description="Retrieve a web page and convert it to text an agent can read",
        read_only=True,
    ),
)


_BY_KEY = {entry.key: entry for entry in DIRECTORY}


def find(key: str) -> Entry | None:
    return _BY_KEY.get(key.strip().lower())


def enabled() -> bool:
    """Whether directory hosts are reachable without the operator allow-list."""
    return get_settings().mcp_directory_enabled


def hosts() -> frozenset[str]:
    """Hosts the directory vouches for. Empty when the directory is switched off."""
    if not enabled():
        return frozenset()
    found = set()
    for entry in DIRECTORY:
        if entry.transport is MCPTransport.HTTP and entry.url:
            host = (urlparse(entry.url).hostname or "").lower()
            if host:
                found.add(host)
    return frozenset(found)


def vouches_for(host: str) -> bool:
    return host.strip().lower() in hosts()


def as_dicts() -> list[dict[str, object]]:
    """The directory as the UI needs it, including why an entry is unusable."""
    from mcp_connect import policy  # noqa: PLC0415 - avoids an import cycle

    stdio_ready = bool(policy.allowed_commands())
    directory_on = enabled()

    listed: list[dict[str, object]] = []
    for entry in DIRECTORY:
        if entry.transport is MCPTransport.STDIO:
            blocked = (
                ""
                if stdio_ready
                else (
                    "This one runs a program on the worker, so an administrator has "
                    "to allow the command first (CWAP_MCP_ALLOWED_COMMANDS)."
                )
            )
        elif directory_on or _allow_listed(entry.url):
            blocked = ""
        else:
            blocked = (
                "Directory servers are switched off on this deployment "
                "(CWAP_MCP_DIRECTORY), so this host needs the outbound allow-list."
            )

        listed.append(
            {
                "key": entry.key,
                "label": entry.label,
                "summary": entry.summary,
                "category": entry.category,
                "transport": entry.transport.value,
                "url": entry.url,
                "command": entry.command,
                "args": list(entry.args),
                "docs_url": entry.docs_url,
                "description": entry.description,
                "read_only": entry.read_only,
                "argument_hint": entry.argument_hint,
                "available": not blocked,
                "blocked_reason": blocked,
                "credentials": [
                    {
                        "name": credential.name,
                        "label": credential.label,
                        "how": credential.how,
                        "required": credential.required,
                        "prefix": credential.prefix,
                    }
                    for credential in entry.credentials
                ],
            }
        )
    return listed


def _allow_listed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    allowed = get_settings().http_allowed_hosts
    return bool(host) and (
        host in allowed or any(host.endswith(f".{entry}") for entry in allowed)
    )


#: The order the UI shows groups in: what people connect first, first.
CATEGORIES = ("Code", "Research", "Operations", "Local")

__all__ = [
    "CATEGORIES",
    "DIRECTORY",
    "Credential",
    "Entry",
    "as_dicts",
    "enabled",
    "find",
    "hosts",
    "vouches_for",
]

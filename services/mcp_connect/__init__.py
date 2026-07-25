"""MCP connectors — reaching systems the platform did not write a connector for.

Named `mcp_connect` rather than `mcp` so it cannot shadow the official `mcp` SDK
package on the import path, which it would otherwise do from inside this service.
"""

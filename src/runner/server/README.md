# server/

Internal HTTP sidecar that backs the use.computer dashboard's "Run with model" / Adhoc / Replay flows. Not part of the cookbook surface — most readers can skip this directory.

It exists here (instead of inside the gateway) because Harbor needs a Python process to host the agents, and this server is what the gateway forwards run requests to.

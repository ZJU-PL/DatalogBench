# The one way out of the agent network.
#
# The agent container sits on an `--internal` docker network, which has no route
# off the host at all. That alone would be too much: the CLI has to reach its
# own model, and a container that cannot do that cannot run the setting. So the
# agent's only egress is this proxy, which sits on both networks and forwards
# CONNECT only to the single API host.
#
# A proxy rather than firewall rules inside the agent container: rules there
# would need NET_ADMIN, and handing that capability to a process running
# model-written shell commands would give away more than the rules buy.
#
# The allowlist is on the CONNECT target, so TLS is not intercepted -- the proxy
# never sees the traffic, only where it is going.

FROM alpine:3.20

RUN apk add --no-cache tinyproxy

# Filter is an allowlist (FilterDefaultDeny), matched against the CONNECT host.
# ConnectPort is limited to 443 so the tunnel cannot be repurposed for another
# service on the same host.
#
# The allowlist itself is NOT baked in. It is mounted at run time from a file the
# harness generates out of the endpoints the agents are actually configured to
# call. Baking one host into the image is what went wrong the first time: the
# image allowed the direct-prompting endpoint while the agents were configured
# against two different ones, and the isolation check passed because it tested
# the host in the image rather than the host the CLI would dial.
RUN printf '%s\n' \
      "Port 8888" \
      "Listen 0.0.0.0" \
      "Timeout 600" \
      "Allow 0.0.0.0/0" \
      "ConnectPort 443" \
      "Filter \"/etc/tinyproxy/allow.txt\"" \
      "FilterDefaultDeny Yes" \
      "FilterType ere" \
      "FilterCaseSensitive No" \
      "LogLevel Notice" \
      > /etc/tinyproxy/tinyproxy.conf \
 && printf '%s\n' "^$^" > /etc/tinyproxy/allow.txt

# The baked allow.txt matches nothing, so an unmounted allowlist denies
# everything rather than falling open.
EXPOSE 8888
CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]

# IP Forwarding Security Analysis — Aspen GPU Nodes

## Date: 2026-02-17

## Summary

`net.ipv4.ip_forward` was enabled on aspen1-4 to allow Docker container ports
(specifically the GPU sidecar on port 8007) to be reachable from external hosts.
This document explains the problem, the fix, the security ramifications, and
recommended hardening steps.

---

## The Problem

Docker containers with published ports were unreachable from any external host.
The GPU sidecar (`gpu-sidecar` container, port 8007) worked locally
(`curl localhost:8007`) but timed out when accessed from MindRouter2 or any
other machine on the network.

### Root Cause

`net.ipv4.ip_forward` was set to `0` (disabled) on all four aspen hosts.

When Docker publishes a container port, it uses DNAT (Destination NAT) in the
kernel's netfilter PREROUTING chain to rewrite the destination address from the
host IP (e.g., `129.101.249.149:8007`) to the container's internal IP (e.g.,
`10.77.0.2:8007`). After this rewrite, the kernel must **forward** the packet
from the external interface (`eno1`) across to the Docker bridge (`docker0`).

With IP forwarding disabled, the kernel silently dropped these packets after
DNAT. No RST, no ICMP unreachable — just a timeout.

Local connections worked because they traverse the loopback interface, which
bypasses the forwarding path entirely.

### Diagnosis Path

1. Confirmed sidecar worked locally on aspen4 (`curl localhost:8007` returned 401)
2. Confirmed connection from aspen1 to aspen4:8007 timed out
3. Confirmed connection from aspen1 to aspen4:8000 (vllm, host-native) worked
4. Inspected full nftables ruleset — DNAT, FORWARD, and DOCKER chains all correct
5. Found `net.ipv4.ip_forward = 0` — the kernel refused to route between interfaces

---

## The Fix Applied

On all four hosts (aspen1, aspen2, aspen3, aspen4):

```bash
# Enable immediately
sudo sysctl -w net.ipv4.ip_forward=1

# Persist across reboots
echo 'net.ipv4.ip_forward = 1' | sudo tee /etc/sysctl.d/99-docker-forward.conf
```

---

## Affected Hosts

| Host   | IP               | Interfaces                        | Sidecar |
|--------|------------------|-----------------------------------|---------|
| aspen1 | 129.101.249.146  | eno1, ibp41s0, docker0            | :8007   |
| aspen2 | (campus subnet)  | eno1, ibp41s0, docker0            | :8007   |
| aspen3 | (campus subnet)  | eno1, ibp41s0, docker0            | :8007   |
| aspen4 | 129.101.249.149  | eno1, ibp41s0, docker0            | :8007   |

---

## Security Ramifications

### What `ip_forward=1` Does

It enables the Linux kernel to act as a **packet router**, forwarding packets
between any of its network interfaces. Without it, packets arriving on one
interface that are destined for a different interface are dropped.

### Risk: Cross-Network Routing

Each aspen host has at least three interfaces:

- **eno1** — campus network (129.101.x.x)
- **ibp41s0** — InfiniBand HPC interconnect (private fabric)
- **docker0** — Docker bridge network (10.77.0.0/16)

With global forwarding enabled, if an attacker on the campus network sends
crafted packets through an aspen host destined for the InfiniBand fabric, the
kernel could theoretically route them across — turning the host into an
unintended bridge between the campus network and the private HPC interconnect.

### Current Mitigations

The nftables FORWARD chain has **policy DROP**:

```
chain FORWARD {
    type filter hook forward priority filter; policy drop;
    jump DOCKER-USER
    jump DOCKER-FORWARD
}
```

Only Docker's DOCKER-FORWARD sub-chains permit forwarding, and only for
packets that have been DNAT'd to container IPs on docker0. In practice:

| Path                          | Result      | Reason                              |
|-------------------------------|-------------|-------------------------------------|
| Campus (eno1) -> Docker       | **Allowed** | Docker DNAT + FORWARD rules match   |
| Campus (eno1) -> InfiniBand   | **Dropped** | No FORWARD rule matches, policy drop|
| InfiniBand -> Campus          | **Dropped** | No FORWARD rule matches, policy drop|
| InfiniBand -> Docker          | **Dropped** | No FORWARD rule matches, policy drop|

### Remaining Concerns

1. **Intentional hardening was overridden.** The fact that `ip_forward=0` was
   set on all hosts suggests a deliberate security policy (system hardening
   script or security team configuration). Docker normally enables this at
   startup — something was actively disabling it.

2. **DOCKER-USER chain is empty.** Docker provides this chain specifically for
   administrators to add custom restrictions on forwarded traffic. Currently it
   permits anything Docker's own rules allow, from any source IP.

3. **Global scope is broader than necessary.** `net.ipv4.ip_forward=1` enables
   forwarding on ALL interfaces, including InfiniBand. A more targeted
   configuration would only enable forwarding on the interfaces Docker needs.

---

## Recommended Hardening

### Option 1: Per-Interface Forwarding (Preferred)

Replace the global setting with interface-specific forwarding. This keeps
InfiniBand isolated from the forwarding path entirely:

```bash
# /etc/sysctl.d/99-docker-forward.conf
# Only enable forwarding on interfaces Docker needs
net.ipv4.ip_forward = 1
net.ipv4.conf.ibp41s0.forwarding = 0
```

Note: `ip_forward=1` is required globally for per-interface settings to take
effect, but setting `ibp41s0.forwarding=0` prevents the InfiniBand interface
from participating in forwarding.

### Option 2: Restrict Sidecar Access via DOCKER-USER

Add nftables rules so only the MindRouter2 production server
(129.101.202.148) can reach container ports:

```bash
# Allow only MindRouter2 server to reach sidecar
sudo nft add rule ip filter DOCKER-USER ip saddr != 129.101.202.148 tcp dport 8007 drop

# To persist, add to a startup script or nftables config file
```

### Option 3: Both (Recommended)

Apply both per-interface forwarding AND DOCKER-USER source restrictions.

### Implementation on Each Host

```bash
# 1. Update sysctl config
cat <<'EOF' | sudo tee /etc/sysctl.d/99-docker-forward.conf
# Enable IP forwarding for Docker container port mapping
net.ipv4.ip_forward = 1
# Disable forwarding on InfiniBand to keep HPC fabric isolated
net.ipv4.conf.ibp41s0.forwarding = 0
EOF
sudo sysctl -p /etc/sysctl.d/99-docker-forward.conf

# 2. Restrict sidecar access to MindRouter2 server only
sudo nft add rule ip filter DOCKER-USER \
    ip saddr != 129.101.202.148 tcp dport 8007 drop
```

---

## Verification

After applying hardening, verify:

```bash
# 1. Sidecar still reachable from MindRouter2 server
#    (run from mindrouter2.insight.uidaho.edu)
curl -H "X-Sidecar-Key: <key>" http://aspen4.hpc.uidaho.edu:8007/health

# 2. Sidecar NOT reachable from random campus hosts
#    (should timeout or be rejected)
curl --connect-timeout 5 http://aspen4.hpc.uidaho.edu:8007/health

# 3. InfiniBand forwarding disabled
sysctl net.ipv4.conf.ibp41s0.forwarding
# Expected: net.ipv4.conf.ibp41s0.forwarding = 0

# 4. DOCKER-USER rule in place
sudo nft list chain ip filter DOCKER-USER
# Expected: ip saddr != 129.101.202.148 tcp dport 8007 drop
```

---

## References

- Docker networking documentation: https://docs.docker.com/network/
- Linux ip_forward: https://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt
- nftables DOCKER-USER chain: https://docs.docker.com/network/packet-filtering-firewalls/

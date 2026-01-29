# Data Collection Guide for ML Training

## Overview

This guide explains how to capture real network traffic using Wireshark for training our SDN traffic classifier. The goal is to collect labeled samples of each priority class (P0-P3) with authentic application signatures.

---

## Priority Classes to Capture

| Priority | Class Name | Target Samples | Duration/Session |
|----------|------------|----------------|------------------|
| **P3** | Banking/Payment | 50-100 flows | 2-3 min per app |
| **P2** | Voice/Video | 50-100 flows | 5 min per app |
| **P1** | Web/Office | 100-200 flows | 5 min browsing |
| **P0** | Bulk/Background | 50-100 flows | 2-3 min per download |

**Total target: 300-500 labeled flows minimum**

---

## Capture Setup

### Wireshark Configuration

1. **Start Wireshark** as administrator/root
2. **Select interface**: Your main network interface (Wi-Fi or Ethernet)
3. **Capture filter** (optional, to reduce noise):
   ```
   tcp port 443 or tcp port 80 or udp port 443 or udp portrange 3478-3479 or udp portrange 16384-32767
   ```
4. **Start capture before** opening each application

### File Naming Convention

Save each capture with this naming pattern:
```
{priority}_{app_name}_{timestamp}.pcapng

Examples:
P3_hdfc_netbanking_20260129_1430.pcapng
P2_google_meet_20260129_1445.pcapng
P1_web_browsing_20260129_1500.pcapng
P0_ubuntu_iso_download_20260129_1515.pcapng
```

---

## P3: Banking/Payment Traffic (Highest Priority)

### What to Capture

Banking transactions require lowest latency and highest reliability.

### Applications to Use

| App/Website | Actions to Perform | Expected Duration |
|-------------|-------------------|-------------------|
| **HDFC NetBanking** | Login, check balance, view statement | 2-3 min |
| **SBI YONO** | Login, check account, mini statement | 2-3 min |
| **ICICI iMobile** | Login, view accounts, fund transfer page | 2-3 min |
| **Google Pay** | Open app, view transactions, check balance | 2 min |
| **PhonePe** | Open app, check history, view offers | 2 min |
| **Paytm** | Login, wallet balance, view passbook | 2 min |
| **Razorpay Dashboard** | Login, view transactions (if you have access) | 2 min |
| **PayPal** | Login, view activity, settings | 2 min |

### Capture Procedure

1. Start Wireshark capture
2. Open browser/app in incognito/fresh session
3. Navigate to banking site and perform login
4. Browse through 3-5 pages (balance, statements, etc.)
5. **DO NOT** perform actual transactions (just viewing)
6. Log out
7. Stop capture and save with `P3_` prefix

### Expected Traffic Patterns

- Short bursts of HTTPS traffic
- Small packet sizes (mostly < 1500 bytes)
- Low inter-arrival times during active use
- TLS 1.2/1.3 with identifiable SNI domains

---

## P2: Voice/Video Traffic (Low Jitter Required)

### What to Capture

Real-time communication requiring consistent low latency.

### Applications to Use

| App/Website | Actions to Perform | Expected Duration |
|-------------|-------------------|-------------------|
| **Google Meet** | Join a test meeting, stay for 5 min | 5 min |
| **Zoom** | Join test meeting or personal room | 5 min |
| **Microsoft Teams** | Start a call (even solo) | 3-5 min |
| **Discord** | Join voice channel, stay connected | 5 min |
| **WhatsApp Web** | Make a voice/video call | 3 min |
| **Slack Huddle** | Start a huddle | 3-5 min |
| **Spotify/YouTube Music** | Stream music (live, not cached) | 3 min |

### Capture Procedure

1. Start Wireshark capture
2. Open video conferencing app
3. Join a meeting (use test meetings or call yourself from another device)
4. Keep camera ON for video traffic
5. Speak occasionally for audio traffic
6. Stay connected for full duration
7. Leave meeting, stop capture, save with `P2_` prefix

### Expected Traffic Patterns

- Steady UDP streams (SRTP/WebRTC)
- Consistent packet sizes (~200-1200 bytes for audio, larger for video)
- Very regular inter-arrival times (20-40ms for audio)
- STUN/TURN traffic on ports 3478, 19302

---

## P1: Web/Office Traffic (Best Effort)

### What to Capture

Normal browsing and office productivity work.

### Applications to Use

| App/Website | Actions to Perform | Expected Duration |
|-------------|-------------------|-------------------|
| **Google Docs** | Open document, type, format text | 3 min |
| **Google Sheets** | Open spreadsheet, edit cells | 3 min |
| **Gmail** | Read emails, compose (don't send) | 3 min |
| **GitHub** | Browse repos, view code, read issues | 3 min |
| **Stack Overflow** | Search, read answers | 2 min |
| **News sites** | CNN, BBC, Times of India - browse | 3 min |
| **Wikipedia** | Read 3-5 articles | 3 min |
| **Reddit** | Browse front page, read posts | 3 min |

### Capture Procedure

1. Start Wireshark capture
2. Open browser (preferably fresh profile)
3. Browse naturally through multiple sites
4. Click links, scroll, read content
5. Open new tabs, switch between them
6. After 5 minutes, stop capture, save with `P1_` prefix

### Expected Traffic Patterns

- Bursty HTTPS traffic
- Mix of small (requests) and large (responses) packets
- Irregular inter-arrival times
- Many different destination IPs/domains

---

## P0: Bulk/Background Traffic (Lowest Priority)

### What to Capture

Large downloads that can be throttled without user impact.

### Applications to Use

| App/Website | Actions to Perform | Expected Duration |
|-------------|-------------------|-------------------|
| **Ubuntu ISO** | Download ubuntu-24.04-desktop-amd64.iso | Until 50-100 MB |
| **Steam** | Download/update a game (pause after 100 MB) | 2-3 min |
| **Windows Update** | Trigger update download (if pending) | 2-3 min |
| **Dropbox/Drive** | Upload/download large folder | 2-3 min |
| **OneDrive** | Sync large files | 2-3 min |
| **apt/yum update** | `sudo apt update && sudo apt upgrade` | Until done |
| **Docker pull** | `docker pull ubuntu:latest` | Until done |
| **npm install** | Large project with many dependencies | 2-3 min |
| **Torrent** (legal) | Linux distro torrent | 2-3 min |

### Capture Procedure

1. Start Wireshark capture
2. Start the download/sync
3. Let it run for 2-3 minutes OR until 50-100 MB
4. Pause/cancel download
5. Stop capture, save with `P0_` prefix

### Expected Traffic Patterns

- Sustained high-bandwidth TCP streams
- Large packets (mostly MTU-sized, ~1500 bytes)
- Very consistent packet flow
- Few unique destinations
- Long flow duration

---

## Post-Capture Checklist

After each capture session, verify:

- [ ] File saved with correct naming convention
- [ ] Capture is at least 1-2 minutes long
- [ ] File size is reasonable (> 1 MB for most captures)
- [ ] No personal/sensitive data accidentally captured

### Organizing Captures

Create this folder structure:
```
captures/
  P3_banking/
    P3_hdfc_netbanking_20260129_1430.pcapng
    P3_sbi_yono_20260129_1435.pcapng
    ...
  P2_voice_video/
    P2_google_meet_20260129_1445.pcapng
    P2_zoom_call_20260129_1500.pcapng
    ...
  P1_web_office/
    P1_web_browsing_20260129_1515.pcapng
    P1_google_docs_20260129_1520.pcapng
    ...
  P0_bulk/
    P0_ubuntu_iso_20260129_1530.pcapng
    P0_steam_download_20260129_1545.pcapng
    ...
```

---

## Exporting from Wireshark

After capturing, export the data for analysis.

### Option 1: Export as CSV (Recommended for Colab)

1. Open the PCAP file in Wireshark
2. Go to **File > Export Packet Dissections > As CSV**
3. Select columns:
   - No. (packet number)
   - Time
   - Source
   - Destination
   - Protocol
   - Length
   - Info
4. Save as `{original_name}.csv`

### Option 2: Keep as PCAP

We have a script that can process PCAP files directly using Python's `scapy` or `pyshark`. Just upload the PCAP files.

### Option 3: TShark Export (Advanced)

For detailed flow-level features:
```bash
tshark -r capture.pcapng -T fields \
  -e frame.time_epoch \
  -e ip.src -e ip.dst \
  -e tcp.srcport -e tcp.dstport \
  -e udp.srcport -e udp.dstport \
  -e ip.proto \
  -e frame.len \
  -e tls.handshake.extensions_server_name \
  -E header=y -E separator=, > capture_flows.csv
```

---

## What Features We'll Extract

From your captures, we'll compute these features per flow:

### Flow Identification
- `src_ip`, `dst_ip` (anonymized)
- `src_port`, `dst_port`
- `protocol` (TCP=6, UDP=17)

### Volume Features
- `packet_count` - Total packets in flow
- `byte_count` - Total bytes transferred
- `duration_sec` - Flow duration in seconds

### Rate Features
- `bytes_per_packet` - Average packet size
- `packets_per_sec` - Packet rate
- `bytes_per_sec` - Throughput

### Statistical Features
- `pkt_len_min` - Minimum packet size
- `pkt_len_max` - Maximum packet size
- `pkt_len_mean` - Mean packet size
- `pkt_len_std` - Std deviation of packet sizes
- `iat_mean` - Mean inter-arrival time
- `iat_std` - Std deviation of inter-arrival time

### Application Features
- `sni_domain` - TLS Server Name Indication (if present)
- `dst_port_category` - Well-known port classification

### Label
- `priority` - P0, P1, P2, P3 (from filename)

---

## Minimum Data Requirements

For reasonable model performance:

| Priority | Minimum Flows | Ideal Flows |
|----------|---------------|-------------|
| P3 | 30 | 100+ |
| P2 | 30 | 100+ |
| P1 | 50 | 200+ |
| P0 | 30 | 100+ |
| **Total** | **140** | **500+** |

### Quality Checklist

- Each class should have samples from **at least 3 different applications**
- Each capture should be **at least 1 minute** long
- Avoid captures with **mixed traffic** (e.g., don't browse web while on video call)
- Keep **one activity per capture** for clean labeling

---

## Upload Instructions

Once you have captured the data:

1. **Zip all PCAP/CSV files** preserving folder structure:
   ```bash
   zip -r sdn_traffic_captures.zip captures/
   ```

2. **Upload to Google Drive** or share directly

3. **In Colab**, we'll:
   - Load all captures
   - Extract flow features automatically
   - Label based on filename prefix (P0/P1/P2/P3)
   - Train and evaluate models

---

## Privacy Notes

- **Anonymize** any captures before sharing (IP addresses will be masked in processing)
- **Avoid** capturing actual banking transactions - just UI navigation
- **Don't** capture passwords, tokens, or sensitive form data
- Wireshark captures TLS-encrypted content as ciphertext (not readable), but SNI domains are visible

---

## Troubleshooting

### "Capture file too small"
- Ensure the application was actually active during capture
- Check that you selected the correct network interface

### "No TLS/SNI visible"
- Some modern browsers use ECH (Encrypted Client Hello)
- Try using Chrome without ECH or Firefox
- The domain will be in DNS queries if not in TLS

### "Too many flows"
- This is fine! More data = better model
- We'll aggregate packets into flows during processing

---

## Next Steps

After you've collected captures:
1. Share the data (zip file or Drive link)
2. We'll process it in Colab notebooks
3. Train classifier and predictor models
4. Export models for SDN orchestrator integration

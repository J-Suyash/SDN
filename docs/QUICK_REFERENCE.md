# Data Collection Quick Reference Card

## Capture Checklist

### Priority P3 - Banking (Capture 50-100 flows)
```
□ HDFC NetBanking - Login, view balance, statements (2-3 min)
□ SBI YONO - Login, account details (2-3 min)  
□ Google Pay - View transactions, balance (2 min)
□ PhonePe - Check history, wallet (2 min)
□ PayPal - Login, view activity (2 min)
```
**Save as**: `P3_appname_YYYYMMDD_HHMM.pcapng`

---

### Priority P2 - Voice/Video (Capture 50-100 flows)
```
□ Google Meet - Join test meeting, camera ON (5 min)
□ Zoom - Test call or personal room (5 min)
□ Discord - Join voice channel (5 min)
□ WhatsApp - Voice/video call (3 min)
□ Spotify - Stream music live (3 min)
```
**Save as**: `P2_appname_YYYYMMDD_HHMM.pcapng`

---

### Priority P1 - Web/Office (Capture 100-200 flows)
```
□ Google Docs - Edit document (3 min)
□ Gmail - Read/compose emails (3 min)
□ GitHub - Browse repos, code (3 min)
□ News sites - Browse CNN, BBC (3 min)
□ Wikipedia - Read articles (3 min)
```
**Save as**: `P1_appname_YYYYMMDD_HHMM.pcapng`

---

### Priority P0 - Bulk Downloads (Capture 50-100 flows)
```
□ Ubuntu ISO - Download ~50-100 MB then pause
□ Steam - Download game update, pause after 100 MB
□ Windows Update - Trigger pending update
□ Docker pull - docker pull ubuntu:latest
□ npm install - Large project
```
**Save as**: `P0_appname_YYYYMMDD_HHMM.pcapng`

---

## Wireshark Setup

1. **Start as admin**: `sudo wireshark`
2. **Select interface**: Wi-Fi or Ethernet (main internet)
3. **Optional filter**: `tcp port 443 or tcp port 80 or udp`
4. **Start capture BEFORE opening app**
5. **Stop capture AFTER closing app**

---

## Folder Structure

```
captures/
├── P3_banking/
│   ├── P3_hdfc_20260129_1430.pcapng
│   └── P3_googlepay_20260129_1445.pcapng
├── P2_voice_video/
│   ├── P2_meet_20260129_1500.pcapng
│   └── P2_zoom_20260129_1515.pcapng
├── P1_web_office/
│   ├── P1_browsing_20260129_1530.pcapng
│   └── P1_gmail_20260129_1545.pcapng
└── P0_bulk/
    ├── P0_ubuntu_iso_20260129_1600.pcapng
    └── P0_steam_20260129_1615.pcapng
```

---

## Export Options

### Option A: Keep as PCAP (Recommended)
Just zip the folder and upload to Colab.

### Option B: Export as CSV
1. File → Export Packet Dissections → As CSV
2. Include: No., Time, Source, Destination, Protocol, Length, Info

---

## Quality Checks

- [ ] Each capture > 1 minute
- [ ] Each file > 1 MB (except P3 banking)
- [ ] At least 3 different apps per priority class
- [ ] No mixed activities (one app per capture)

---

## After Collection

1. **Zip all captures**:
   ```bash
   zip -r sdn_captures.zip captures/
   ```

2. **Upload to Google Drive**

3. **Open Colab and run notebooks**:
   - `01_data_exploration.py` → Convert to .ipynb and run
   - `02_train_classifier.py` → Train priority classifier
   - `03_train_predictor.py` → Train congestion predictor

---

## Target: 300-500 total flows across all classes

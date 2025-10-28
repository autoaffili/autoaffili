# Privacy Policy

Effective: October 2025

This Privacy Policy explains how Autoaffili ("we," "us," or "our") processes audio and related information when you use the plugin, helper, and service to generate transcripts and post links to Twitch chat.

By using the Service, you agree to this Policy. If you do not agree, do not use the Service.

## Summary
- Audio is processed locally to produce transcripts stored at `%LOCALAPPDATA%\autoaffili\logs\diagnostics.csv`.
- We do not retain raw audio. We do not sell, share or train models with your personal information.
- Transcript text (not audio) is sent to our backend to detect product mentions and create short links.

## Information We Process
- Microphone and desktop audio: Captured on locally on device to generate transcripts. Processed in memory and not stored. No audio rentention. No audio transmission to servers. To be clear, the service operates continually while enabled and begins operation once "Start Streaming" is clicked and ends upon closure of OBS or clicking "Stop Streaming" in OBS. 
- Transcripts and metadata: Text output, timestamps, and detection results. Transcript text is sent to our backend over HTTPS for product‑mention classification and link generation.
- Account and auth data: Twitch OAuth tokens and channel login for posting messages and to donate any commission generated; stored locally at `%LOCALAPPDATA%\autoaffili\twitch_tokens.json` and Twitch channel name is also stored in our backend.  
- Technical data: Local logs (e.g., `diagnostics.csv`, ASR debug JSON, `errors.log`) and status files in `%LOCALAPPDATA%\autoaffili\`.

## How We Use Information
- Provide the Service: Convert audio to text locally; detect product mentions using our backend; generate a short link; post to your Twitch chat.
- Communications: Errors, auth via backend google cloud queries. 

## Where Data Lives
- Audio: Processed locally and discarded after transcription.
- Transcripts/logs: Stored locally under `%LOCALAPPDATA%\autoaffili\logs\`. By default, diagnostics are reset on plugin startup; you can delete them at any time.
- Tokens: Stored locally at `%LOCALAPPDATA%\autoaffili\twitch_tokens.json` and persist until you delete them or revoke access. They will never be stored on servers.

## Third‑Party Services
- Autoaffili backend: Receives transcript text (not audio) to classify product mentions and to create short links. Hosted in the United States by default.
- Link shortener: Our backend will use Geniuslink to produce short, localized URLs.
- Twitch: We post messages to your Twitch chat using your channel and tokens.
Your use of these services is subject to their own terms and privacy policies.

## Legal Bases (EEA/UK)
Where GDPR/UK GDPR applies, we rely on:
- Performance of a contract: To provide the Service you request.
- Legitimate interests: To maintain security, prevent abuse, and improve the Service.
- Consent: Where required (e.g., microphone access, posting to chat).

## Retention
- Audio: Not retained.
- Transcripts/logs: Local files are reset on plugin startup by default and may be deleted by you at any time.
- Tokens: Persist locally until you delete them or revoke access.

## Your Choices and Rights
- Controls: Enable/disable capture; choose whether to run the service.
- Delete: Remove the local app folder (`%LOCALAPPDATA%\autoaffili\`) to clear logs and tokens. Also, uninstall program locally. 
- Access and deletion: Most data remains under your control locally. Where applicable (EEA/UK/California), you may request access, correction, deletion, or restriction. We do not sell personal information.

## Security
We apply reasonable administrative and organizational safeguards. OAuth tokens are stored locally and used only to perform authorized actions. No method of transmission or storage is 100% secure.

## Children
The Service is not directed to children under 13 (or the minimum age in your jurisdiction). Do not use the Service to capture others’ voices without their knowledge and consent.

## International Transfers
If you use our backend or host the Service outside your country, your data (transcript text) may be processed in other jurisdictions (e.g., United States) with different data protection laws.

## Changes
We may update this Policy. We will indicate the effective date at the top and, where required, notify you of material changes.

## Contact
Questions or requests: autoaffilibusiness@gmail.com

This document is provided for general informational purposes and is not legal advice. You should consult your own legal counsel to tailor this policy to your deployment, data flows, and jurisdiction.


# Whisper-small (fine-tuned) on RunPod Serverless: one language per endpoint

1. In `Dockerfile`, set MODEL_ID (your HF repo) and LANGUAGE (yoruba / hausa / igbo).
2. Put the 4 files at the root of a GitHub repo (e.g. `whisper-yoruba`).
3. RunPod -> Serverless -> New Endpoint -> GitHub Repo -> pick the repo.
4. GPU: 16 GB is plenty. Active workers 0, container disk 15 GB, FlashBoot on.
5. Test in the Requests tab:
   {"input": {"audio_url": "https://raw.githubusercontent.com/USER/REPO/main/clip.mp3"}}

Private model? Remove the snapshot_download line in the Dockerfile and add an
environment variable HF_TOKEN on the endpoint.

If you see an error about the language token, add env var FORCE_LANGUAGE=0 on the endpoint.

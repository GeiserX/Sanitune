# Security Policy

Sanitune processes audio files locally. While it doesn't handle authentication or user accounts in self-hosted mode, we still take security seriously.

## Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please use GitHub's private vulnerability reporting:

1. Go to https://github.com/GeiserX/Sanitune/security/advisories
2. Click "Report a vulnerability"
3. Fill out the form with details

We will respond within **48 hours** and work with you to understand and address the issue.

### What to Include

- Type of issue (e.g., path traversal, arbitrary file read, command injection)
- Full paths of affected source files
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact assessment and potential attack scenarios

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | Current release    |

Only the latest version receives security updates. We recommend always running the latest version.

## Security Considerations

### File Processing

- Audio files are processed locally and are never uploaded by the Phase 1 pipeline
- Uploaded files are validated for format before processing
- Temporary files are cleaned up after processing completes
- File paths are sanitized to prevent path traversal attacks
- Phase 1 writes output as `.wav`

### Docker Security

- Container runs as non-root user
- The current Docker image is a CPU-only CLI container
- Core audio processing does not require outbound network access after dependencies/models are available
- Optional lyrics lookup and first-run model downloads may require outbound network access
- Volumes are used for input/output only

### API Keys (Optional)

- Optional lyrics lookup can use a user-provided `GENIUS_API_KEY`
- Keys are never stored persistently by Sanitune
- Keys are passed only to the selected third-party lyrics provider when that feature is explicitly used

### Privacy Notes

- By default, audio stays local
- If you pass `--artist` and `--title` with lyrics extras installed, Sanitune may send song metadata to syncedlyrics and/or Genius
- Sanitune does not send audio content to those lyrics providers

## Contact

For security questions that aren't vulnerabilities, open a GitHub issue.

---

*Last updated: April 2026*

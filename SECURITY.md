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

- Audio files are processed locally — no data leaves the machine
- Uploaded files are validated for format before processing
- Temporary files are cleaned up after processing completes
- File paths are sanitized to prevent path traversal attacks

### Docker Security

- Container runs as non-root user
- No network access required for core processing (models downloaded at startup)
- Volumes are used for input/output only

### API Keys (Optional)

- AI contextual replacement requires a user-provided LLM API key
- Keys are never stored persistently — used for the current session only
- Keys are never logged or transmitted to any third party

## Contact

For security questions that aren't vulnerabilities, open a GitHub issue.

---

*Last updated: April 2026*

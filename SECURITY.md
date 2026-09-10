# Security Policy

BudgetTrace is designed to keep secrets out of the repository and to fail fast on missing live configuration.

Public artifact writers apply recursive key- and value-based redaction before serialization. The value scanner intentionally catches common credential shapes such as `Bearer` tokens, URL userinfo, OpenAI-style `sk-` keys, and PEM private-key headers. This may redact some harmless strings that look like credentials; prefer the false positive over publishing a secret.

If you discover a security issue, do not paste credentials, private data, or sensitive tokens into issues or pull requests. Report the problem through the project maintainer's private contact path and include only the minimum reproduction details needed to understand the issue.

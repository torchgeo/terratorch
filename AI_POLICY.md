# AI Policy

This policy covers the use of generative AI tools (coding assistants, chat
models, agents) when contributing to TerraTorch — code, tests, documentation,
configuration, examples, issues and pull requests.

## AI use is allowed

Using AI tools is explicitly permitted. You do not need permission, and you do
not need to justify it. We care about the result, not about how it was
produced.

## The author is responsible for every line

Whoever opens a pull request owns it completely. That holds for every single
line, regardless of whether a human or a model wrote it.

Concretely, this means:

- **You must understand what you submit.** If you cannot explain why a line is
  there, what it does, and what happens when it fails, it is not ready.
- **You must verify it.** Run the tests, run the linters, read the diff. "The
  model said it works" is not verification.
- **You own the bugs.** A defect introduced with AI assistance is your defect.
  There is no "the tool wrote that" in review, in a bug report, or in a
  post-mortem.
- **You own the licensing.** Check that generated code, comments and assets do
  not carry incompatible licenses or copied text from an unknown source. The
  repository's license obligations apply unchanged.
- **You own the security.** Do not paste secrets, credentials, customer data or
  unpublished internal material into third-party AI services. Review generated
  code for the same classes of issues we review human code for — unsafe
  deserialization, path traversal, injection, unbounded resource use.

## Review

AI-assisted contributions are reviewed exactly like any other contribution.
Reviewers may ask you to explain any part of the change, and may ask for it to
be rewritten. "This came from a model" is not an answer to a review comment and
does not lower the bar.

Disclosure of AI assistance is welcome but not required. Accountability is
required.

## Scope

This policy applies to everyone contributing to the repository, including
maintainers. It does not replace the [Security Policy](SECURITY.md) or the
project license; where they overlap, the stricter rule applies.

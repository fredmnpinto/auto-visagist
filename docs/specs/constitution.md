# Constitution — Project Principles

This document records the core principles that guide development of the
Facial Visagism Analysis System. All contributors and agents are expected
to follow them.

## 1. Spec-Driven Development

The `functional-spec.md` is the single source of truth for what the system
does. Every implementation decision must trace back to a requirement in
the spec. Feature creep (anything outside Must/Should) is not allowed
unless explicitly approved.

## 2. Test Coverage ≥85%

All new code must maintain or improve line and branch coverage. The project
target is **≥85%** (measured by `pytest --cov`). Code that drops coverage
below the threshold should not be merged.

## 3. Test-Driven Development (TDD)

Write tests before implementation whenever practical. At minimum:

- Write the test that defines the expected behaviour first.
- Confirm it fails (red).
- Implement the feature until the test passes (green).
- Refactor as needed while keeping tests green.

This applies to all new features and bug fixes. Exceptions are allowed for
experimental or exploratory code, but the tests must be added before such
code is merged.

## 4. No Secrets in the Repository

Never commit credentials, API keys, model files, or other sensitive data.
Use environment variables or configuration files excluded by `.gitignore`.

## 5. Document External Sources

Any code adapted from external sources (Stack Overflow, blog posts,
libraries, etc.) must include a comment referencing the original source.
This is a requirement of the MSc assignment rules.

# Engineering Guide

All content in this guide is synthetic demo data for the Agentic Search project.

## 1. Branching and Code Review

Work happens on short-lived feature branches created from main. A branch that lives longer
than five working days must be rebased on main daily to limit merge conflicts.

Every pull request needs two approvals before it can merge, and at least one approval must
come from the owning team. Pull requests larger than 400 changed lines are sent back for
splitting unless the author marks them as a mechanical refactor.

Reviewers are expected to give first feedback within one business day. A pull request with
no review activity for three days is escalated to the team lead.

## 2. Testing Standards

Unit test line coverage must stay at or above 80 percent for every service. The build fails
if coverage drops more than 2 percentage points compared with main.

Integration tests run against ephemeral containers, never against shared staging data.
A flaky test must be quarantined within one business day and fixed or deleted within two
weeks; quarantined tests are reviewed every sprint.

## 3. Release Process

Production releases go out on Tuesdays and Thursdays between 14:00 and 16:00 Eastern Time.
Releases outside that window require sign-off from the on-call engineer and the team lead.

A release freeze applies during the last two business days of each quarter, and during any
active SEV1 incident. Hotfixes are exempt from the freeze but still need two approvals.

Every release must include a rollback plan, and the rollback must be verified in staging
before the deployment starts. Feature flags are the preferred rollout mechanism: new
features start at 5 percent of traffic and ramp over at least three days.

## 4. On-Call and Incidents

The on-call rotation lasts seven days and starts every Monday at 10:00 Eastern Time. Each
service has a primary and a secondary on-call engineer.

Pages must be acknowledged within fifteen minutes. If the primary does not acknowledge, the
page escalates to the secondary after ten more minutes, and then to the team lead.

Incidents are classified by severity. SEV1 means a full outage or data loss and requires a
response within thirty minutes plus a status update every thirty minutes. SEV2 means major
degradation for a subset of users and requires a response within two hours. SEV3 covers
minor issues handled during business hours.

A written postmortem is required for every SEV1 and SEV2 incident and must be published
within five business days. Postmortems are blameless and must list concrete action items
with named owners and due dates.

## 5. Dependencies and Security

Dependency scans run nightly. Critical vulnerabilities must be patched within seven days,
high severity within thirty days, and medium severity within ninety days.

Secrets are never committed to the repository. Any secret that reaches a branch must be
rotated immediately, even if the commit is removed afterwards.

Production database access requires a ticket, is granted for a maximum of four hours, and
is always logged and reviewed monthly.

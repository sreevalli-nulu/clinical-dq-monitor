# Clinical data quality report (explanations)

Flags come from rules and statistics. Claude only wrote the wording. Severity and numbers are from code.

## [HIGH] Fatal Not Serious for subject 01-701-1211
*RULE_FINDING - site 701 - subject 01-701-1211 - explanation source: template*

A fatal adverse event is not marked as serious. Here: outcome FATAL but serious = N (AE SUDDEN DEATH).

**Why it matters:** Records that break this rule can distort the safety or efficacy picture if they are not corrected.

**What to check:**
- Compare the record with the source document.
- Consider the common innocent explanation: By definition a fatal event is serious, so this is nearly always a missing or wrong flag in the record.

**Caution:** This is a lead for a reviewer, not a conclusion that the record is wrong.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [HIGH] Fatal Not Serious for subject 01-704-1445
*RULE_FINDING - site 704 - subject 01-704-1445 - explanation source: template*

A fatal adverse event is not marked as serious. Here: outcome FATAL but serious = N (AE COMPLETED SUICIDE).

**Why it matters:** Records that break this rule can distort the safety or efficacy picture if they are not corrected.

**What to check:**
- Compare the record with the source document.
- Consider the common innocent explanation: By definition a fatal event is serious, so this is nearly always a missing or wrong flag in the record.

**Caution:** This is a lead for a reviewer, not a conclusion that the record is wrong.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [HIGH] Fatal Not Serious for subject 01-710-1083
*RULE_FINDING - site 710 - subject 01-710-1083 - explanation source: template*

A fatal adverse event is not marked as serious. Here: outcome FATAL but serious = N (AE MYOCARDIAL INFARCTION).

**Why it matters:** Records that break this rule can distort the safety or efficacy picture if they are not corrected.

**What to check:**
- Compare the record with the source document.
- Consider the common innocent explanation: By definition a fatal event is serious, so this is nearly always a missing or wrong flag in the record.

**Caution:** This is a lead for a reviewer, not a conclusion that the record is wrong.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [HIGH] Visit After Death for subject 01-710-1083
*RULE_FINDING - site 710 - subject 01-710-1083 - explanation source: template*

A visit is dated after the subject's recorded death. Here: visit on 2013-08-03, death on 2013-08-02 (visit WEEK 2).

**Why it matters:** Records that break this rule can distort the safety or efficacy picture if they are not corrected.

**What to check:**
- Compare the record with the source document.
- Consider the common innocent explanation: Either the visit date or the death date may be mistyped (often off by a day), or the visit sits under the wrong subject.

**Caution:** This is a lead for a reviewer, not a conclusion that the record is wrong.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [ALERT] Site 713: higher than other sites (digit 0 5 share)
*SITE_SIGNAL - site 713 - explanation source: template*

Average share of a patient's BP/pulse readings ending in 0 or 5 (digit preference; only HIGH is a concern). Site 713 is at 71.9 percent against 34.1 percent in the other sites (z-score 8.98, 9 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Staff or devices may round readings; check which device is used and who takes the measurement.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [ALERT] Site 705: higher than other sites (digit 0 5 share)
*SITE_SIGNAL - site 705 - explanation source: template*

Average share of a patient's BP/pulse readings ending in 0 or 5 (digit preference; only HIGH is a concern). Site 705 is at 62.1 percent against 33.6 percent in the other sites (z-score 8.88, 16 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Staff or devices may round readings; check which device is used and who takes the measurement.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [ALERT] Site 708: higher than other sites (digit 0 5 share)
*SITE_SIGNAL - site 708 - explanation source: template*

Average share of a patient's BP/pulse readings ending in 0 or 5 (digit preference; only HIGH is a concern). Site 708 is at 44.5 percent against 34.5 percent in the other sites (z-score 3.46, 25 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Staff or devices may round readings; check which device is used and who takes the measurement.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [ALERT] Site 704: higher than other sites (out of window patient rate)
*SITE_SIGNAL - site 704 - explanation source: template*

Dosed subjects with at least one out-of-window visit. Site 704 is at 76.0 percent against 43.2 percent in the other sites (z-score 3.31, 25 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Travel distance, staffing or holidays can delay visits; check whether one or two patients drive the rate.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [WATCH] Site 717: higher than other sites (dq flag rate)
*SITE_SIGNAL - site 717 - explanation source: template*

Dosed subjects with a HIGH/MEDIUM rule finding. Site 717 is at 42.9 percent against 10.9 percent in the other sites (z-score 2.71, 7 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: More rule findings per patient points to entry problems at the site; small sites move a lot with a few patients.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [WATCH] Site 703: lower than other sites (any ae rate)
*SITE_SIGNAL - site 703 - explanation source: template*

Dosed subjects with at least one adverse event (low = possible under-reporting). Site 703 is at 66.7 percent against 87.3 percent in the other sites (z-score -2.63, 18 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: A low rate can mean under-reporting, but also a different mix of placebo and active patients (not adjusted for here) or a small sample.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [WATCH] Site 701: lower than other sites (out of window patient rate)
*SITE_SIGNAL - site 701 - explanation source: template*

Dosed subjects with at least one out-of-window visit. Site 701 is at 31.7 percent against 49.3 percent in the other sites (z-score -2.25, 41 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Travel distance, staffing or holidays can delay visits; check whether one or two patients drive the rate.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [WATCH] Site 713: lower than other sites (dropout rate)
*SITE_SIGNAL - site 713 - explanation source: template*

Dosed subjects who discontinued. Site 713 is at 22.2 percent against 58.0 percent in the other sites (z-score -2.17, 9 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Dropout depends on how sick patients are and on the mix of study arms; a small site can drift by chance.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

## [WATCH] Site 704: higher than other sites (dropout rate)
*SITE_SIGNAL - site 704 - explanation source: template*

Dosed subjects who discontinued. Site 704 is at 76.0 percent against 54.6 percent in the other sites (z-score 2.15, 25 patients).

**Why it matters:** A site that differs a lot from the others can point to a data entry or conduct problem worth a closer look.

**What to check:**
- Review the records behind this rate at the site.
- Consider the common innocent explanation: Dropout depends on how sick patients are and on the mix of study arms; a small site can drift by chance.

**Caution:** 65 site checks were scored, so about 3.0 flags are expected by chance alone.

> Fallback used: AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'API key is invalid.'}, 'request_id': None}

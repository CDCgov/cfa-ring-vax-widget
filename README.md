# Ring vaccination widget

## Overview

This repo contains code for investigating the potential efficacy of ring
vaccination for a disease interactively via a [streamlit](https://streamlit.io/)
app.

## Model description

- Disease progression: susceptible, exposed (i.e., will go onto infection),
  infectious, recovered
- Individuals make contacts on some network, with some effective (i.e.,
  infection transmitting) contact rate. (Contact rates and effective contact
  probability could be separated in parameter input, but they are perfectly
  confounded in the model.)
- Each infected individual can be detected. E.g., infected people might notice
  their own symptoms.
- When an infected person is detected, contact tracing and isolation are
  initiated.
  - Contact tracing
    - Contact tracing can detect infections among contacts, contacts of
      contacts, etc.
    - Contact tracing has different performance characteristics for each "ring."
      (The most straightforward choice would be to make second ring performance
      equivalent to first ring, i.e., time from detecting infected contact to
      infected contact-of-contact is identical to time from detecting index
      infection to infected contact.)
  - Isolation reduces transmission by some factor (which might be 100%) for that
    individual.
- Input parameters/assumptions for this model
  - Latent period $t_\mathrm{latent}$ distribution (time from contact to onset
    of infectiousness)
  - Infectious period $t_\mathrm{inf}$ distribution
  - Basic reproductive number $R_0$ (i.e., mean number of secondary infections
    per index infection in the absence of intervention)
    - Note: this defines some mean infectious rate $R_0 / E[t_\mathrm{inf}]$
    - Assume that this infectious rate is constant over the period of
      infectiousness. E.g., if perfectly effective isolation is implemented
      halfway through an individual's infectious period, that halves their
      number of expected secondary infections.
    - Some distribution of number of secondary infections around $R_0$ (e.g.,
      Poisson)
    - Assume that the number of secondary infections is uncorrelated across
      individuals (i.e., there is some kind of homogeneity in the contact
      network).
  - Per-infection detection (aka "passive" detection, in which individuals
    identify their own symptoms)
    - % of infections identified in this way (in the absence of contact tracing)
    - Distribution of times from exposure to detection (or, equivalently, from
      exposure to symptom onset and from symptom onset to detection)
  - Contact tracing
    - % of first ring (contacts) identified
    - Distribution of times from index exposure to contact identification
    - Time-varying performance of contact tracing (e.g., a simple assumption
      would be that all infected contacts immediately stop infecting, i.e., that
      the diagnostic test can identify infections with 100% sensitivity
      immediately after exposure and isolation is 100% effective)
    - % of second ring (contacts of contacts) identified, distribution of times,
      etc.
    - Third ring, etc.
- Implementation/initialization
  - Seed a single infection (e.g., exposed via travel)
  - **_Open question_**: How to deal with multiple rings?
    - What are the network implications of treating undetected infections as new
      index infections (with independent but identically distributed number of
      secondary infections)?
    - How hard would it be to simulate contacts on a simple network?
- Output/viz
  - Some discrete realizations, showing the timelines of events for individuals
    (e.g., how the different disease state periods line up in time)
  - Distribution of number of undetected
- Assumptions of note
  - Assuming independence is conservative: clustering helps you

## Project Admins

- Scott Olesen (CDC/CFA) <ulp7@cdc.gov>
- Andy Magee (CDC/CFA) <rzg0@cdc.gov>
- Paige Miller (CDC/CFA) <yub1@cdc.gov>

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public
health related projects in support of the
[CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not
hosted by the CDC, but is a third party website used by CDC and its partners to
share information and collaborate on software. CDC use of GitHub does not imply
an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is
in the public domain within the United States, and copyright and related rights
in the work worldwide are waived through the
[CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication.
By submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice

This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or
modify it under the terms of the Apache Software License version 2, or (at your
option) any later version.

This source code in this repository is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache Software
License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md) and
[Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit
[http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by
[forking](https://help.github.com/articles/fork-a-repo) and submitting a pull
request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law,
including but not limited to the Federal Records Act, and may be archived. Learn
more at
[http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

# Ring vaccination widget

## Overview

This repo contains code for investigating the potential efficacy of ring
vaccination for a disease interactively via a [streamlit](https://streamlit.io/)
app.

## Model description
The model is a branching process with the following description.
Note that we the model does not track (and indeed has no notion of) susceptibles.

- Disease progression:
  - Exposed/latent (i.e., will go on to infection)
  - Infectious
  - Recovered
- Infection process
  - There are a finite number of generations
    - Default is 3: index infection (i.e., generation 0), generation 1
      (contacts), generation 2 (contacts of contacts), and generation 3
      (potential escaping infections)
  - Number and timing of infections generated by infected people
    - Assumed to be a Poisson process: draw inter-infection times from
      $\mathrm{Exp}(1/\lambda)$ until infectious period ends or infection is
      detected
      - N.B.: All "contacts" here are effective contacts, i.e., encounters that
        result in transmission
- Passive detection (i.e., self-detection)
  - Each infection has an independent probability of potential passive detection
    - N.B.: To actually _be_ passively detected, the detection must occur at a time before recovery and before being detected by other means.
  - If passively detected, detection occurs at some time distribution since
    exposure.
    - Start with Dirac delta distribution (i.e., all passively-detected
      infections are detected at some fixed delay after exposure)
    - N.B.: This assumes that progression of infectiousness and symptoms are
      independent. We could not say that, e.g., symptoms begin immediately upon
      onset of infectiousness, and the delay to self-detection is some time
      after that.
  - N.B.: There is no assumption that index case is passively detected. If the
    index case does not self-detect, this is not an automatic fail, since they
    might not infect anyone, or their infectees might self-detect.
- Contact tracing (i.e., active detection)
  - Every detected infection (whether passive or active) automatically triggers contact tracing
  - Contact tracing has an independent probability of detecting each infection
    caused by the detected infection
    - N.B.: To actually _be_ actively detected, the detection must occur at a time before recovery and before being detected by other means.
    - N.B.: Contact tracing goes only forward and only one generation. For
      example, say index infects A infects B infects C, and the index is not
      detected, but A is passively detected. Then this creates an chance to
      actively detect B, but not the index or C (although C might be detected if
      the detection of A leads to contact tracing that detects B that in turn
      leads to contact tracing that detects C).
    - N.B.: "Detection" here means detection _and_ successful intervention. We
      do not separately model the detected infection's probability of divulging
      contact information, the ability of public health to find that contact, or
      the probability of that contact to comply with quarantine/isolation.
  - There is a distribution of times between triggering detection and contact
    tracing completion. Start with Dirac delta.
- Input parameters/assumptions for this model
  - Latent period $t_\mathrm{latent}$ distribution (time from contact to onset
    of infectiousness). Start with Dirac delta.
  - Infectious period $t_\mathrm{inf}$ distribution. Start with Dirac delta.
  - Infectious rate. Start with identical for all people.
  - Passive detection probability and delay distribution: Dirac delta
  - Active detection probability and delay distribution: Dirac delta
- Initialization: Seed a single infection (e.g., exposed via travel)
- Outputs
  - High-level aggregate summaries of all simulations
  - Visual of history of each individual simulation.

### Scope
The scope of this repo is a model which is a (1) branching processes where (2) the offspring distribution (including the times at which subsequent infections are caused) for any infection depends only on the history of the process up until the time at which they are infected.
For practical purposes, this most likely means the scope can be considered to be density-independent branching processes.

The model implemented herein may be iterated upon subject to preserving this basic structure.
For example, any model which requires tracking susceptibles or a network structure is out of scope.
But replacing Dirac delta distributions with other probability distributions for disease history would be in scope.

## Analysis

- Define a "successful" simulation as one with zero 3rd-generation infections

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

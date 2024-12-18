# Ring Vaccination

Ring vaccination is a public health strategy to mitigate the spread of an infectious disease by vaccinating individuals who have been exposed to the disease. In a ring vaccination strategy, after a case has been confirmed, health officials try to identify the patient's close contacts and vaccinate them. Ring vaccination also includes identifying and vaccinating the contacts of those contacts to form a protective "ring" of immunity around infected individuals.  This strategy relies on a few key factors:

- short delays between case identification and diagnosis
- effective isolation of diagnosed cases
- short turn around time for contact tracing the contacts of a case
- contact tracing a high fraction of close contacts
- quick vaccination of those contacts before they become infectious
- post-exposure vaccination is effective at preventing transmission
- contact tracing and vaccinating a high fraction of contacts of contacts

In contrast with pro-active vaccination of individuals based on their prior known risk factors, ring vaccination is a form of reactive vaccination.

In the real world, ring vaccination can be an effective strategy for both controlling transmission and preventing disease from developing in contacts. This widget focuses on simulating a simplified ring vaccination intervention using a branching process model to investigate it's potential to control transmission in a population. This means that some of the above factors have been simplified in the model and that we are not explicitly modeling the benefit of reduced disease burden separate from reduced transmission (see our [README](../README.md) for more model details).

## Use of ring vaccination in the past

[Ring vaccination has previously been used as one part of the successful strategy to globally eradicate smallpox, which was declared officially eradicated by the World Health Assembly in 1980. Accounts of the program indicate that the strategy relied on intensive public health surveillance to identify cases early, isolate them, and interview cases to identify their close contacts](https://www.nfid.org/the-triumph-of-science-the-incredible-story-of-smallpox-eradication/). Those contacts were then notified of their exposure, had their health monitored, and were vaccinated. Contacts were also interviewed to identify their contacts, and those contacts of contacts were vaccinated as well.

Ring vaccination has also been part of the strategy for limiting the spread of Ebola virus outbreaks. In 2015, a [ring vaccination trial of an Ebola virus vaccine (rVSV-ZEBOV Ebola vaccine) began in Guinea](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(16)32621-6/fulltext). Interim trial results yielded [evidence of high vaccine efficacy for this approach](https://doi.org/10.1177/17407745211073594). As a result of those interim results, the World Health Organization Strategic Advisory Group of Experts on Immunization (SAGE) endorsed a ring vaccination deployment of the vaccine, as well as vaccinating other at-risk groups, such as health care and front-line workers. Since then, ring vaccination has been used to limit spread in other Ebola outbreaks in addition to other targeted vaccine strategies, including during the [2019 outbreak in DRC](https://www.who.int/news/item/23-09-2019-second-ebola-vaccine-to-complement-ring-vaccination-given-green-light-in-drc#:~:text=The%20main%20vaccination%20strategy%20used,transfer%20skills%20to%20the%20region.).

## Contact tracing

Part of a successful ring vaccination strategy is highly effective contact tracing. Estimates of contact tracing vary widely between outbreaks and settings. In some cases, high contact tracing can be achieved when symptoms are unambiguous and conditions are right for health officials and the public to work together to identify and share their contacts.

In contrast with airborne diseases and diseases spread through direct contact with bodily fluids or skin-to-skin contact, contact tracing can be less successful for sexually transmitted diseases. [Stigma associated with having a sexually transmitted disease or the fear of disclosing sexual partners can hinder contact tracing](doi: 10.2105/AJPH.2022.306842). As a result, we expect that contact tracing for a sexually transmitted disease in most settings would be less successful than for Ebola or smallpox.

Contact tracing can also be effective at controlling transmission when combined with other interventions other than ring vaccination. [Simulation studies on the control of COVID-19 spread prior to the widespread availability of vaccines](https://doi.org/10.1038/s41467-021-23276-9) demonstrate that high contact tracing combined with high capacity for testing, short test delay times, and high rates of quarantine can be effective at controlling transmission.

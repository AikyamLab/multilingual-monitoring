# The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages

<p align="center">
  <a href="https://multilingual-cot-monitoring.github.io/"><img src="https://img.shields.io/badge/Website-Project_Page-1f6feb?style=flat&labelColor=222222"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=flat&labelColor=222222"></a>
</p>

<p align="center"><em>by</em></p>

<table align="center">
  <tr>
    <td align="center"><b>Eric Onyame</b><br>University of Virginia</td>
    <td align="center"><b>Runtao Zhou</b><br>University of Virginia</td>
    <td align="center"><b>Kowshik Thopalli</b><br>Lawrence Livermore<br>National Laboratory</td>
    <td align="center"><b>Bhavya Kailkhura</b><br>Lawrence Livermore<br>National Laboratory</td>
    <td align="center"><b>Chirag Agarwal</b><br>University of Virginia</td>
  </tr>
</table>

<p align="center"><b>Corresponding author:</b> Eric Onyame</p>

<br>

This repository hosts the codebase for **The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages**, the first large-scale evaluation of CoT monitorability under linguistic distribution shift. Below, we provide an overview of the project along with key implementation details.

## Overview

Chain-of-thought (CoT) monitoring has been proposed as a promising safety mechanism for detecting misaligned behavior in large language models — yet its reliability has remained largely unexplored beyond English and across diverse model families. We present the first large-scale evaluation of CoT monitorability across **13 typologically diverse languages** and **seven frontier model families** (16 models total, spanning 8B–120B parameters), using adversarial-hint evaluations that require explicit intermediate computation together with logit-lens analysis of internal answer-token probabilities.

We find that CoT unfaithfulness persists consistently across languages and hint types, with an **average deception rate of 95.9%**. Frontier models systematically engage in strategic manipulation — including answer-switching, post-hoc rationalization, and **procedural exploitation of hints** — making external monitors struggle to detect deception. Logit-lens analysis further reveals that models often commit to the misaligned cue in their latent activations **within the first 15% of generation**, even when the CoT appears faithful. Strikingly, these deceptive patterns **saturate at 100% in low-resource languages**, revealing fundamental limitations in current CoT-based oversight.

Our results show that CoT monitoring is fundamentally fragile under linguistic distribution shift, providing a substantially weaker safety signal than what English-only studies suggest. These findings underscore an urgent need to develop robust CoT monitors and to accelerate research into white-box monitoring techniques — especially to improve CoT monitorability in mid- and low-resource languages.

For full technical details and experiments, see the [project website](https://multilingual-cot-monitoring.github.io/).
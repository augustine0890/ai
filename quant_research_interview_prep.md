# Quant Research Interview Prep Pack

Prepared on 2026-03-19 for the eight open-ended interview prompts you provided.

These are not standard puzzle questions. They are research-design questions. A strong answer has to do five things quickly:

1. Reframe the problem in precise technical terms.
2. Offer one sane baseline and one scalable or production-ready approach.
3. Explain tradeoffs, assumptions, and failure modes.
4. State how you would validate the method.
5. Show that you can implement a first version without overcomplicating it.

---

## Part I. What Each Prompt Is Actually Testing

| Prompt | Hidden subjects | What the interviewer is looking for |
| --- | --- | --- |
| 1. Low-memory robust regression | Robust statistics, online learning, stochastic optimization, matrix sketching | You can preserve robustness without a batch algorithm that stores all data |
| 2. Computers recognizing "obviously strange" markets | Anomaly detection, regime detection, change points, labeling, representation learning | You can turn vague human pattern recognition into a measurable ML problem |
| 3. Non-static, dependent observations | Time series, dependence, heteroskedasticity, drift, block bootstrap, state-space models | You know IID is often false and can replace it with the right machinery |
| 4. NP-complete trading-system optimization | Combinatorial optimization, relaxations, approximation algorithms, heuristics, integer programming | You know when exact optimality is impossible and how to get near-optimal solutions safely |
| 5. Shared computation across many related integrals | Numerical analysis, variance reduction, reduced-order modeling, offline-online decomposition | You look for structure before burning compute |
| 6. Measuring market impact of our own trading | Market microstructure, transaction cost analysis, causal confounding, execution modeling | You can define the counterfactual and measure costs without fooling yourself |
| 7. Combining returns and options to infer joint distributions | Risk-neutral vs physical measures, option-implied densities, dependence modeling, copulas, entropy methods | You understand what options identify and what historical data identifies |
| 8. Filtering Twitter to market-relevant content | Streaming NLP, weak supervision, entity matching, text classification, drift, evaluation under class imbalance | You can build a low-latency relevance system, not just an offline classifier |

---

## Part II. Universal Answer Framework For Questions Like These

Use this exact structure in interviews.

### A. Reframe the question
- "The core issue is not X, it is Y under constraint Z."
- Example: "This is really robust estimation under memory constraints, so I would think in terms of online M-estimation or sketch-and-solve methods."

### B. Give a baseline
- Mention the textbook batch solution first.
- Then say why it fails here.
- Example: "A batch Huber or IRLS fit is the baseline, but it is memory-heavy if the dataset is too large."

### C. Give a scalable method
- Propose 1 to 3 realistic alternatives.
- Mention which one you would try first and why.

### D. Explain validation
- What metric?
- What benchmark?
- What ablation?
- What backtest or holdout design?

### E. Mention failure modes
- Drift
- selection bias
- leakage
- nonstationarity
- latency
- numerical instability
- overfitting
- hidden confounding

### F. End with implementation
- "I would prototype this in Python first with `numpy`, `scipy`, `statsmodels`, `scikit-learn`, or `cvxpy`, then move bottlenecks to C++ if latency mattered."

---

## Part III. Preparation Resources By Interview Prompt

## 1. Low-Memory Robust Regression

### What you should be able to say
- Classical robust regression uses M-estimators such as Huber or Tukey losses.
- If memory is the bottleneck, move from batch fitting to online or minibatch optimization.
- For linear models, stochastic gradient methods with robust losses are the first practical tool.
- If dimensionality is large, sketching or coreset ideas can compress the problem before solving.
- Validation should compare against a batch robust baseline on smaller subsamples.

### Core topics
- M-estimators
- Huber loss
- stochastic gradient descent
- online convex optimization
- matrix sketching
- subspace embeddings
- coresets

### Primary resources
- [HuberRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
- [Stochastic Gradient Descent user guide](https://sklearn.org/stable/modules/sgd.html)
- [Online Learning chapter from Understanding Machine Learning](https://www.cambridge.org/core/books/understanding-machine-learning/online-learning/A43CEEF3F7F18953B592C983087C8A36)
- [River: online machine learning in Python](https://github.com/online-ml/river)
- [Sketching as a Tool for Numerical Linear Algebra](https://research.ibm.com/publications/sketching-as-a-tool-for-numerical-linear-algebra)
- [Woodruff lecture notes on sketching](https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/allLectures.pdf)

### Implementation resources
- `sklearn.linear_model.HuberRegressor`
- `sklearn.linear_model.SGDRegressor`
- `river` for streaming updates

### Drills
- Implement linear regression with `loss="huber"` in `SGDRegressor` and compare against batch OLS and batch Huber.
- Write a note answering: when would you prefer online Huber over sketch-and-solve?
- Explain how you would monitor convergence if you never revisit the whole dataset.

### Interview follow-up questions to practice
- How would you choose the Huber threshold?
- When does sketching preserve the solution quality poorly?
- What if the outliers are adversarial rather than random?

---

## 2. Getting A Computer To Recognize "Strange" Markets

### What you should be able to say
- This can be posed as anomaly detection, regime detection, or change-point detection depending on what labels exist.
- If humans can point to examples, try supervised or weakly supervised classification.
- If labels are scarce, combine unsupervised anomaly scores with regime segmentation and human review.
- Feature design matters more than model novelty: returns, realized vol, order imbalance, spread, depth, correlation breakdown, options skew, news features.
- Evaluation should reward early detection, not just overall classification accuracy.

### Core topics
- anomaly detection
- novelty detection
- change-point detection
- regime switching
- Markov switching models
- hidden Markov models
- weak supervision
- human-in-the-loop labeling

### Primary resources
- [Ruptures: change point detection in Python](https://github.com/deepcharles/ruptures)
- [Markov switching models in statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html)
- [OneClassSVM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [SGDOneClassSVM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html)
- [LocalOutlierFactor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [River anomaly and drift components](https://github.com/online-ml/river)
- [Change-Point Detection and Its Modern Applications](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-041124-044143)

### Implementation resources
- `ruptures`
- `statsmodels` Markov switching
- `sklearn` novelty and outlier detection tools
- `river` for streaming drift detection

### Drills
- Build a toy regime detector on returns and realized volatility.
- Compare change-point detection against a two-regime Markov model.
- Create a false-positive analysis: what market conditions look "strange" statistically but are just illiquid or news-driven?

### Interview follow-up questions to practice
- What labels would you ask traders for?
- How do you distinguish a true new regime from a temporary outlier?
- How would you detect that your anomaly detector has itself drifted?

---

## 3. When IID And Stationarity Fail

### What you should be able to say
- First, diagnose which assumption fails: serial dependence, heteroskedasticity, structural breaks, changing regimes, cross-sectional dependence, or concept drift.
- Replace IID tools with time-series-aware methods.
- For inference, use HAC or block bootstrap rather than naive standard errors.
- For dynamics, use ARIMA, VAR, GARCH, state-space, or online learning depending on the use case.
- For unstable environments, use adaptive or rolling estimation instead of one global fit.

### Core topics
- stationarity
- autocorrelation
- heteroskedasticity
- HAC / Newey-West
- block bootstrap
- ARIMA / VAR
- GARCH
- state-space models
- Kalman filtering
- concept drift

### Primary resources
- [MIT OpenCourseWare Time Series Analysis notes](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/resources/lecture-notes/)
- [statsmodels state-space documentation](https://www.statsmodels.org/stable/statespace.html)
- [Markov switching example in statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html)
- [Newey-West HAC covariance in statsmodels](https://www.statsmodels.org/v0.13.5/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)
- [arch documentation](https://arch.readthedocs.io/)
- [Understanding Machine Learning: online learning](https://www.cambridge.org/core/books/understanding-machine-learning/online-learning/A43CEEF3F7F18953B592C983087C8A36)
- [River drift examples](https://riverml.xyz/dev/)

### Implementation resources
- `statsmodels`
- `arch`
- `river`

### Drills
- Estimate a regression with naive standard errors and then with HAC. Explain the difference.
- Compare rolling OLS, expanding OLS, and a Kalman-filter state-space model.
- Run a block bootstrap and explain why plain bootstrap is wrong here.

### Interview follow-up questions to practice
- When is rolling estimation better than a state-space model?
- What breaks if you use random train-test splits on time series?
- How would you detect a structural break before re-estimating everything?

---

## 4. Near-Optimal Solutions To NP-Complete Problems

### What you should be able to say
- Start by checking whether the exact subproblem is really NP-hard or whether the firm's instance has exploitable structure.
- If it is NP-hard, use exact methods only for small instances to create a benchmark.
- Then use relaxations, decomposition, approximation algorithms, or local search for scale.
- Always report optimality gaps or benchmark against exact solutions on smaller problems.
- In trading systems, latency and stability often matter more than provable global optimality.

### Core topics
- NP-hardness
- approximation algorithms
- linear relaxations
- Lagrangian relaxation
- branch-and-bound
- mixed-integer programming
- local search
- greedy heuristics
- decomposition

### Primary resources
- [Stanford Approximation Algorithms course](https://web.stanford.edu/class/msande319/Approximation%20Algorithm/index.html)
- [OR-Tools introduction](https://developers.google.com/optimization/introduction)
- [OR-Tools get started guides](https://developers.google.com/optimization/introduction/get_started)
- [CVXPY project](https://www.cvxpy.org/)
- [OSQP documentation](https://osqp.org/docs/)
- [NetworkX maximum-weight matching](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html)

### Implementation resources
- `ortools` CP-SAT and MIP
- `cvxpy`
- `osqp`
- `networkx`

### Drills
- Take a combinatorial toy problem and solve it three ways: greedy, MILP, local search.
- Measure runtime and objective gap as the instance size grows.
- Practice saying: "I would first see whether our production objective admits a tractable relaxation."

### Interview follow-up questions to practice
- When do you trust a greedy heuristic?
- What would you log in production to detect optimizer failure?
- How do you keep a near-optimal solution stable from one run to the next?

---

## 5. Sharing Expensive Work Across Many Related Integrals

### What you should be able to say
- Look for common structure before optimizing numerics.
- If the functions are related, reuse quadrature nodes, random numbers, basis expansions, or surrogate approximations.
- In Monte Carlo settings, use common random numbers and control variates.
- In many-query problems, reduced-basis and offline-online decomposition are the key idea.
- If the problem is hierarchical, multilevel Monte Carlo may reduce cost sharply.

### Core topics
- numerical quadrature
- change of variables
- control variates
- common random numbers
- importance sampling
- multilevel Monte Carlo
- reduced-basis methods
- empirical interpolation
- sparse grids

### Primary resources
- [Multilevel Monte Carlo methods by Mike Giles](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/C5AF9A57ED8FF8FDF08074C1071C5511/S096249291500001Xa.pdf/multilevel_monte_carlo_methods.pdf)
- [Use of control variates in Monte Carlo estimation](https://academic.oup.com/jrsssc/article-abstract/31/2/125/6985259)
- [Reduced basis methods for many-query problems](https://www.nas.nasa.gov/pubs/ams/2023/12-07-23.html)
- [Empirical interpolation method paper](https://www.mit.edu/~cuongng/publication/pub1/)
- [Sparse Grids and Applications](https://link.springer.com/book/10.1007/978-3-030-81362-8)

### Implementation resources
- `scipy.integrate`
- `numpy` with cached nodes and weights
- custom Monte Carlo code with common random seeds

### Drills
- Estimate several related expectations with and without common random numbers.
- Build a control variate for a toy integral where one related integral has a closed form.
- Explain what "offline-online decomposition" means without using PDE jargon.

### Interview follow-up questions to practice
- When would you prefer a surrogate to a quadrature method?
- What if the related functions are only loosely related?
- How do you know shared randomness is reducing variance rather than masking bugs?

---

## 6. Measuring The Market Impact Of Our Own Trading

### What you should be able to say
- Start from the counterfactual: compared with what price path?
- Decompose execution costs into spread, temporary impact, permanent impact, timing, and opportunity cost.
- Implementation shortfall is the basic accounting framework, but measuring causal impact is harder because orders are endogenous.
- Condition on participation rate, urgency, volatility, liquidity, and time of day.
- Use metaorder-level analysis where possible rather than isolated fills.

### Core topics
- implementation shortfall
- temporary vs permanent impact
- participation rate
- market microstructure
- optimal execution
- square-root impact law
- confounding and selection bias

### Primary resources
- [Optimal execution of portfolio transactions](https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions)
- [Optimal Execution: A Review](https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2161588)
- [Impact is not just volatility](https://econpapers.repec.org/RePEc:arx:papers:1905.04569)
- [The double square-root law](https://ideas.repec.org/p/arx/papers/2502.16246.html)
- [The risk of falling short: implementation shortfall variance in portfolio construction](https://www.tandfonline.com/doi/abs/10.1080/1351847X.2025.2558117)

### Implementation resources
- execution-cost decomposition notebooks
- panel regressions or nonparametric fits by ADV, participation, volatility bucket

### Drills
- Define your own implementation shortfall decomposition on a sample order log.
- Fit impact as a function of order size and participation.
- Practice explaining why observed slippage is not equal to causal self-impact.

### Interview follow-up questions to practice
- How do you separate your own impact from market drift?
- How would hidden liquidity change your estimate?
- Why might short-horizon impact estimates be biased?

---

## 7. Combining Historical Returns And Options Data For Joint Distributions

### What you should be able to say
- Options identify risk-neutral information, not directly the physical distribution.
- Historical returns inform realized dynamics and real-world dependence.
- So this is a measure-combination problem, not just a curve-fitting problem.
- A good answer mentions no-arbitrage smoothing, extracting marginal risk-neutral densities, then combining marginals and dependence using a parametric model, copula, state-space model, or entropy method.
- You should be explicit about what is identified by data and what is imposed by modeling assumptions.

### Core topics
- risk-neutral vs physical measure
- state-price density
- Breeden-Litzenberger
- implied volatility surface
- density recovery
- copulas
- entropy pooling
- calibration under no-arbitrage

### Primary resources
- [Prices of State-Contingent Claims Implicit in Option Prices](https://www.gsb.stanford.edu/faculty-research/working-papers/prices-state-contingent-claims-implicit-option-prices)
- [Nonparametric Risk Management and Implied Risk Aversion](https://www.nber.org/papers/w6130)
- [Simple and reliable way to compute option-based risk-neutral distributions](https://fedinprint.org/item/fednsr/12565)
- [Risk-neutral systemic risk indicators](https://fedinprint.org/item/fednsr/12857)
- [Fully Flexible Views: Theory and Practice](https://econpapers.repec.org/RePEc:arx:papers:1012.2848)
- [Closed-form transformations from risk-neutral to real-world distributions](https://www.sciencedirect.com/science/article/pii/S0378426607000258)

### Implementation resources
- surface smoothing and density extraction code
- copula calibration tools
- entropy reweighting notebooks

### Drills
- Recover a risk-neutral marginal density from an option smile.
- Compare it with a historical density estimate.
- Write a memo on what additional assumptions are needed to infer a joint physical distribution from options plus returns.

### Interview follow-up questions to practice
- What parts of the joint distribution are weakly identified?
- Why can two models fit option prices and imply very different real-world tails?
- How would you enforce no-arbitrage while fitting the surface?

---

## 8. Efficiently Filtering Twitter Feeds To Market-Relevant Content

### What you should be able to say
- Start with a pipeline, not a single model.
- Stage 1 is cheap retrieval and hard filtering.
- Stage 2 is entity recognition and normalization.
- Stage 3 is relevance classification and deduplication.
- Stage 4 is novelty scoring or event aggregation.
- Because labels drift and positives are rare, use weak supervision, active learning, and online evaluation.

### Core topics
- streaming ingestion
- rule-based filtering
- NER and entity linking
- text classification
- weak supervision
- online learning
- class imbalance
- concept drift
- deduplication and ranking

### Primary resources
- [X filtered stream documentation](https://docs.x.com/x-api/posts/filtered-stream/introduction)
- [Legacy filtered stream overview](https://developer.x.com/en/docs/twitter-api/filtered-stream-overview)
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)
- [Speech and Language Processing draft](https://web.stanford.edu/~jurafsky/slp3/)
- [spaCy TextCategorizer](https://spacy.io/api/textcategorizer/)
- [spaCy EntityRuler](https://spacy.io/api/entityruler)
- [spaCy Matcher](https://spacy.io/api/matcher/)
- [Out-of-core text classification in scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html)
- [HashingVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
- [Classification of text documents using sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Implementation resources
- `HashingVectorizer`
- `spaCy`
- `sklearn` partial-fit models
- online feedback loop for relabeling

### Drills
- Build a lightweight pipeline with cashtag and issuer filters, then a relevance classifier.
- Measure precision at top-k, not just accuracy.
- Add an entity-linking step and evaluate how much it improves precision.

### Interview follow-up questions to practice
- How would you handle sarcasm or rumor propagation?
- How do you avoid overfitting to yesterday's market narratives?
- Which errors matter more: false positives or false negatives?

---

## Part IV. Cross-Cutting Resource Stack

These are the few resources worth revisiting repeatedly across all eight prompts.

### Statistics, ML, and online learning
- [Understanding Machine Learning](https://www.cambridge.org/core/books/understanding-machine-learning/contents/076FB5C9121E120E7B206E1774DC114D)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [River](https://github.com/online-ml/river)

### Time series and finance
- [MIT Time Series Analysis notes](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/resources/lecture-notes/)
- [statsmodels state-space](https://www.statsmodels.org/stable/statespace.html)
- [arch documentation](https://arch.readthedocs.io/)

### Optimization
- [Stanford Approximation Algorithms](https://web.stanford.edu/class/msande319/Approximation%20Algorithm/index.html)
- [OR-Tools](https://developers.google.com/optimization)
- [CVXPY](https://www.cvxpy.org/)
- [OSQP](https://osqp.org/docs/)

### Market microstructure and execution
- [Optimal execution of portfolio transactions](https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions)
- [Optimal Execution: A Review](https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2161588)

### Options-implied distributions
- [Breeden-Litzenberger at Stanford GSB](https://www.gsb.stanford.edu/faculty-research/working-papers/prices-state-contingent-claims-implicit-option-prices)
- [NBER paper by Ait-Sahalia and Lo](https://www.nber.org/papers/w6130)
- [NY Fed risk-neutral density note](https://fedinprint.org/item/fednsr/12565)

### NLP
- [CS224N](https://web.stanford.edu/class/cs224n/)
- [Jurafsky and Martin SLP draft](https://web.stanford.edu/~jurafsky/slp3/)
- [spaCy API](https://spacy.io/api/)

---

## Part V. Intensive 12-Week Study Plan

Assumption: 20 to 25 hours per week. This is interview prep for open-ended quant research prompts, not general quant prep.

### Week 1. Robust estimation baseline
- Learn M-estimators, Huber loss, influence of outliers.
- Implement OLS, Huber, and LAD on synthetic contaminated data.
- Deliverable: 2-page note on why robust regression is not enough if memory is the true bottleneck.

### Week 2. Online and low-memory estimation
- Study online learning, SGD, `partial_fit`, and sketching.
- Build a streaming regression experiment.
- Deliverable: notebook comparing batch Huber, SGD-Huber, and a compressed/sketched approach.

### Week 3. Anomalies, regimes, and change points
- Study change-point detection, Markov switching, and outlier detection.
- Build one anomaly detector and one regime detector on market features.
- Deliverable: memo on which formulation best matches "obvious to humans" events.

### Week 4. Broken IID assumptions
- Study HAC, block bootstrap, unit roots, breaks, and state-space models.
- Run a rolling-vs-expanding-vs-state-space comparison.
- Deliverable: write-up on why naive cross-validation is wrong on time series.

### Week 5. Combinatorial optimization
- Study relaxations, branch-and-bound, greedy, local search, and approximation algorithms.
- Solve a toy scheduling or assignment problem with OR-Tools and a heuristic.
- Deliverable: runtime-vs-quality comparison table.

### Week 6. Numerical integration and variance reduction
- Study control variates, common random numbers, importance sampling, MLMC.
- Build two estimators for related integrals and compare variance and runtime.
- Deliverable: short note on how you would share expensive work across a family of functions.

### Week 7. Market impact and execution
- Study implementation shortfall, Almgren-Chriss, temporary and permanent impact.
- Build a toy cost decomposition and impact curve.
- Deliverable: note on why measuring self-impact is a causal problem.

### Week 8. Options-implied densities
- Study risk-neutral density extraction and physical-vs-risk-neutral distinctions.
- Recover an implied marginal density from option data or a synthetic surface.
- Deliverable: explain which parts come from no-arbitrage and which parts come from model assumptions.

### Week 9. Joint distributions and dependence
- Study copulas, entropy pooling, and combining options with returns.
- Build a toy two-asset joint distribution using historical dependence plus option-implied marginals.
- Deliverable: memo on the identification limits of the problem.

### Week 10. Streaming NLP
- Build a lightweight relevance filter for social text.
- Use hard filters, entity normalization, and one classifier.
- Deliverable: precision-at-top-k dashboard and error taxonomy.

### Week 11. Full mock-answer week
- For each of the eight prompts, write a 90-second answer and a 5-minute answer.
- Record yourself or speak out loud.
- Force yourself to mention tradeoffs and validation.
- Deliverable: one-page answer sheet per prompt.

### Week 12. Integration and polish
- Revisit weak areas.
- Build two capstone mini-projects from the list below.
- Run mock interviews with timing pressure.
- Deliverable: final interview packet.

---

## Part VI. Capstone Mini-Projects

Do at least two.

### Project A. Streaming robust regression
- Dataset: synthetic plus one real market panel
- Goal: compare batch robust, online robust, and sketched approaches
- Output: notebook plus memo

### Project B. Regime detection engine
- Dataset: market returns, realized vol, spreads, skew
- Goal: detect breaks and label market regimes
- Output: regime timeline plus false-positive analysis

### Project C. Execution cost analyzer
- Dataset: synthetic order and fill log or your own simulated execution data
- Goal: compute implementation shortfall, fit impact curves, test conditioning variables
- Output: TCA-style report

### Project D. Option-implied plus historical joint density
- Dataset: synthetic option surface plus return history
- Goal: recover marginals, combine with dependence assumptions, simulate joint scenarios
- Output: assumptions memo and validation plots

### Project E. Market-relevant Twitter filter
- Dataset: curated tweet sample or other news-like text stream
- Goal: build low-latency relevance scoring with entity normalization
- Output: confusion analysis and precision-at-k report

---

## Part VII. Mock Interview Checklist

Before answering, silently check:

- Did I define the problem precisely?
- Did I name a baseline?
- Did I say why the baseline fails?
- Did I propose a practical scalable method?
- Did I mention evaluation?
- Did I mention at least one failure mode?
- Did I say how I would prototype it?

If you miss two of these, your answer will sound underdeveloped even if the ideas are good.

---

## Part VIII. Recommended Python Stack

- `numpy`
- `scipy`
- `pandas`
- `statsmodels`
- `arch`
- `scikit-learn`
- `river`
- `cvxpy`
- `ortools`
- `networkx`
- `spacy`

Only move to C++ after you can explain the statistical design clearly in Python.

---

## Part IX. What A Strong Candidate Sounds Like

A strong candidate does not sound like:

- "I would probably try a neural network."
- "I would use AI to detect patterns."
- "I would backtest a few models and pick the best."

A strong candidate sounds like:

- "I would start with a simple baseline that isolates the core issue."
- "The main risk here is confounding, not just model fit."
- "Options give me risk-neutral information, so I still need an assumption to get the physical distribution."
- "I would validate on smaller exact instances first so I can measure the quality gap of the heuristic."
- "I would use a streaming or sketch-based estimator because the batch robust method is memory-bound."

---

## Part X. Highest-Value Practice Habit

For each of the eight prompts, do this three times:

1. Write a 6-line answer.
2. Expand it into a 2-minute spoken answer.
3. Add one paragraph on how you would validate the method.

That last part is where most candidates are weak.

---

## Source Notes

I prioritized official course pages, official library documentation, and canonical papers or reviews rather than blog-style summaries. Where a classic paper is paywalled, I linked to an official bibliographic page or a reputable working-paper version when available.

# Dynamic Delta Hedging Under Transaction Costs

A quantitative research framework for comparing dynamic hedge policies for a European option on futures under transaction costs and downside-risk constraints.

Selling an option creates an exposure that changes as the futures price moves. Delta hedging can reduce that exposure, but every rebalance has an execution cost. Hedging more aggressively can therefore reduce residual risk while making the position more expensive to manage.

The project studies how a market maker can choose a practical hedge policy when both sides of that trade-off matter. Twenty-five policies are tested under the same simulated market conditions, filtered through ex-ante risk limits, compared through cost and downside risk, stress-tested, and eventually reduced to one locked policy for a finite-capital realized campaign.

---

## Core model

The option is a European call on futures priced with the Black–76 model (Black, 1976).

For futures price $F$, strike $K$, remaining maturity $\tau$, risk-free rate $r$ and volatility $\sigma$,

```math
C(F,\tau)
=
e^{-r\tau}
\left[
F\Phi(d_1)-K\Phi(d_2)
\right]
```

where

```math
d_1
=
\frac{
\ln(F/K)+\tfrac{1}{2}\sigma^2\tau
}{
\sigma\sqrt{\tau}
},
\qquad
d_2=d_1-\sigma\sqrt{\tau}
```

The corresponding Black–76 call delta with respect to the futures price is

```math
\Delta_{\mathrm{call}}
=
e^{-r\tau}\Phi(d_1)
```

where $\Phi(\cdot)$ is the standard normal cumulative distribution function.

All 25 hedge policies use the same Black–76 pricing model and the same theoretical delta target. What changes is the rule that decides when, and by how much, the actual futures hedge should move.

### Futures transaction costs

The futures market is represented by a proportional bid–ask spread around the mid-price.

If the hedge position changes by $\Delta q_t$ contracts at futures price $F_t$, the contract-level cost of crossing a full proportional spread $s_F$ is

```math
C_t
=
|\Delta q_t|
F_t
\frac{s_F}{2}
M
```

where $M$ is the contract multiplier.

The hedge problem is therefore a direct trade-off between reducing delta mismatch and paying to rebalance.

---

## Hedge policies

The Research universe contains 25 strategies across five hedging families.

| Family | Main idea | Parameter values |
|---|---|---|
| Fixed Interval | Rebalance on a fixed time schedule | 1, 2, 5, 10, 21 steps |
| Fixed Band | Keep the hedge inside a constant no-trade region around delta | 0.01, 0.02, 0.03, 0.05, 0.10 |
| Whalley-Wilmott-inspired | Use an adaptive no-trade region based on local option risk and transaction costs | $\lambda=$ 0.25, 0.50, 1.00, 2.00, 4.00 |
| Delta Tolerance | Rebalance fully when the hedge deviates sufficiently from delta | 0.01, 0.02, 0.03, 0.05, 0.10 |
| Asset Tolerance | Rebalance after a sufficiently large futures-price move | 1.50%, 2.25%, 3.50%, 5.00%, 7.00% |

### Fixed Interval

The hedge is reset fully to the current Black–76 delta every $m$ simulation steps and remains unchanged between scheduled rebalances.

```math
q_t^{\mathrm{new}}
=
\begin{cases}
\Delta_t, & n \bmod m = 0,\\
q_t, & \text{otherwise}
\end{cases}
```

Smaller values of $m$ produce more frequent rebalancing.

### Fixed Band

A constant no-trade region is placed around the current delta.

```math
L_t=\max(0,\Delta_t-b),
\qquad
U_t=\min(1,\Delta_t+b)
```

If the current hedge position $q_t$ falls outside this region, it moves only to the nearest boundary. If it remains inside the region, no trade is made.

A wider band accepts more delta deviation in exchange for less trading.

### Whalley-Wilmott-inspired

The Whalley-Wilmott-inspired family uses the same nearest-boundary idea, but the width of the no-trade region changes with the option's local risk.

The adapted half-width is

```math
h_t
=
\left(
\frac{
3cF_t e^{-r\tau}
}{
2\lambda
}
\right)^{1/3}
|\Gamma_t|^{2/3}
```

where $c$ is one half of the full proportional futures bid–ask spread, $\Gamma_t$ is Black–76 gamma and $\lambda$ controls risk aversion.

Higher $\lambda$ produces a narrower no-trade region and therefore more aggressive rebalancing.

The original Whalley-Wilmott analysis concerns options hedged with the underlying stock. This implementation keeps the functional form but adapts it to options on futures using the futures price together with Black–76 delta and gamma. It is therefore treated as Whalley-Wilmott-inspired rather than as an exact futures-specific optimum.

### Delta Tolerance

The hedge is left unchanged until its deviation from the current delta exceeds a threshold $\varepsilon$.

```math
q_t^{\mathrm{new}}
=
\begin{cases}
\Delta_t, & |\Delta_t-q_t|>\varepsilon,\\
q_t, & \text{otherwise}
\end{cases}
```

Unlike the Fixed Band strategy, crossing the threshold triggers a full rebalance back to delta.

### Asset Tolerance

The trigger is based on movement in the futures price rather than directly on delta.

If $F_t^{\mathrm{last}}$ is the futures price at the most recent hedge trade,

```math
q_t^{\mathrm{new}}
=
\begin{cases}
\Delta_t,
&
\left|
\frac{
F_t-F_t^{\mathrm{last}}
}{
F_t^{\mathrm{last}}
}
\right|>a,
\\
q_t,
&
\text{otherwise}
\end{cases}
```

A larger value of $a$ allows the futures price to move further before another full delta rebalance is triggered.

---

## Research workflow

The project treats hedge selection as a decision problem rather than simply choosing the strategy with the highest average P&L.

1. **Price the option and calculate its Greeks with Black–76.**  
   Every policy begins from the same model value and theoretical delta target.

2. **Compare 25 hedge policies on common Monte Carlo paths.**  
   Each policy sees the same underlying market paths so that differences are driven by the hedge rule rather than different simulated market luck.

3. **Apply an ex-ante Desk Mandate.**  
   Strategies that exceed predefined limits on P&L dispersion, downside tail loss or expected hedging cost are removed.

4. **Compare the surviving cost–risk trade-offs.**  
   Qualified policies are placed on a Cost–Expected Shortfall efficient frontier.

5. **Test frontier stability.**  
   Bootstrap resampling checks whether frontier membership survives changes in the sampled Research paths.

6. **Run focused paired comparisons.**  
   Serious candidates are compared path by path using the same simulated markets.

7. **Stress realized-volatility assumptions.**  
   The pricing and hedge model remains fixed while the volatility generating the futures paths is moved below and above the Research volatility.

8. **Select and lock one policy.**  
   The final choice is made from the Research evidence without replacing the trade-off with a composite score.

The selected strategy is then carried into one sequential realized trading campaign.

---

## Desk Mandate

The Desk Mandate is an eligibility filter rather than a ranking system.

A strategy must satisfy limits on terminal P&L dispersion, Expected Shortfall and expected hedging cost.

```math
\sigma_P \le L_\sigma,
\qquad
ES_{5\%} \ge L_{ES},
\qquad
\mathbb{E}[C] \le L_C
```

For interpretation in the frontier analysis, downside risk is written as positive 5% Tail Loss:

```math
\mathrm{TailLoss}_{5\%}
=
-ES_{5\%}(\mathrm{P\&L})
```

The mandate limits are entered as percentages of the Black–76 fair contract value so that the constraints scale with the size of the option exposure.

A policy that violates one of the limits is excluded even if it performs well on another metric.

---

## Quote economics

Black–76 gives the theoretical value of the option, but theoretical value alone does not determine the price at which the position is economically attractive to sell.

Different hedge policies produce different expected hedging economics.

The Research Monte Carlo therefore begins with zero option spread and evaluates each policy relative to Black–76 fair value. For strategies that pass the Desk Mandate, the resulting mean Research P&L is translated into a strategy-specific break-even and recommended ask.

If $V_{\mathrm{fair}}$ is the fair contract value, $\mathbb{E}[P_T]$ is the mean terminal Research P&L and $\Pi^*$ is the desired expected terminal profit,

```math
A_{\mathrm{recommended}}
=
V_{\mathrm{fair}}
+
\left(
\Pi^*-\mathbb{E}[P_T]
\right)e^{-rT}
```

The additional initial premium compounds at the risk-free rate until expiry.

When the target expected terminal profit is zero, the recommended ask is the Research break-even ask.

The quote is determined before deployment. Realized campaign outcomes are never allowed to feed back into the price at which the trade would originally have been accepted.

---

## Cost–Expected Shortfall frontier

Among Desk-Mandate survivors, the primary comparison uses expected futures transaction cost and positive downside Tail Loss.

Both are minimized.

A strategy is dominated when another qualified strategy has both lower expected hedging cost and lower downside tail loss.

The policies that remain form the Cost–Expected Shortfall efficient frontier.

A strategy being on the frontier does not mean that it is automatically the best choice. It means that one objective cannot be improved without giving something up in the other.

The final decision is therefore left explicit rather than being replaced by an arbitrary combined score.

---

## Bootstrap frontier stability

An efficient frontier estimated from one Monte Carlo sample can depend on the particular paths that happened to be drawn.

To examine this, each bootstrap replication resamples the Research path indices and applies the same resampled indices to every qualified strategy.

The Cost–Expected Shortfall frontier is then rebuilt.

For strategy $i$,

```math
\mathrm{Stability}_i
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}
\left\{
i
\text{ is efficient in bootstrap } b
\right\}
```

High stability means that frontier membership repeatedly survives resampling of the available Research paths.

It is not interpreted as the probability that the strategy is truly optimal.

---

## Focused paired comparisons

The frontier compares aggregate cost and downside-risk characteristics. Paired analysis asks whether two serious candidates behave differently on the same underlying paths.

For strategies $A$ and $B$, the pathwise P&L difference is

```math
D_i
=
P_i^{(A)}
-
P_i^{(B)}
```

Because both strategies face the same Research path, much of the shared Monte Carlo variation cancels from the comparison.

If the confidence interval around the paired difference includes zero, the result is treated as statistically indistinguishable rather than forcing a winner.

---

## Realized-volatility sensitivity

The final Research check examines what happens when realized volatility differs from the volatility used to price and hedge the option.

Black–76 pricing and Greeks continue to use the fixed Research volatility while only the volatility generating the futures paths changes across

```math
0.75\sigma_{\mathrm{research}},
\qquad
1.00\sigma_{\mathrm{research}},
\qquad
1.25\sigma_{\mathrm{research}}
```

This deliberately creates an implied-versus-realized volatility mismatch without giving the hedge policy information from the future.

The same standardized Brownian shocks are reused across strategies and scenarios, while the futures execution spread remains unchanged.

---

## Frictionless convergence validation

The hedging engine is checked independently in:

[`frictionless_convergence_validation.ipynb`](frictionless_convergence_validation.ipynb)

The validation removes transaction costs and rebalances the hedge at every available point in the time grid.

As the number of rebalancing steps $N$ increases, discrete hedging error should contract toward the continuous-hedging limit.

The first-order benchmark is

```math
\sigma_{\mathrm{P\&L}}
\propto
N^{-1/2}
```

The notebook checks that:

- terminal P&L dispersion falls as the hedge grid becomes finer,
- mean terminal P&L remains statistically compatible with zero,
- the empirical log–log convergence slope approaches the theoretical value of $-1/2$.

The convergence check is kept separate from the strategy-selection workflow so that numerical validation of the hedging engine is not mixed with the economic comparison of transaction-cost policies.

---

## Final strategy selection

The Research process deliberately does not manufacture a unique winner.

The Desk Mandate removes unacceptable strategies, while the frontier, bootstrap analysis, paired comparisons and volatility sensitivity describe the trade-offs among the survivors.

The final policy is selected manually from the Desk-Mandate-qualified set.

Once selected, its Research quote and Research downside-loss reference are locked before deployment.

The realized campaign cannot revise the earlier Research decision using hindsight.

---

## One realized trading campaign

After the policy is locked, the project observes one sequential finite-capital campaign rather than introducing another Monte Carlo selection layer.

Each campaign trade is a complete option episode:

1. a new option is sold,
2. the futures path evolves from inception to expiry,
3. the selected policy dynamically manages the futures hedge,
4. the option settles,
5. any residual hedge is closed,
6. terminal trade P&L is added to capital before the next trade begins.

Each trade begins from the same standardized option setup, while its realized volatility and futures path are generated independently of strategy execution.

For trade $j$,

```math
\sigma_{\mathrm{realized},j}
\sim
\mathrm{Triangular}
\left(
0.75\sigma_{\mathrm{research}},
\sigma_{\mathrm{research}},
1.25\sigma_{\mathrm{research}}
\right)
```

Black–76 pricing and Greeks continue to use the original Research volatility.

### Position sizing

The selected strategy's positive Research downside-loss reference is

```math
L
=
-ES_{5\%}^{\mathrm{Research}}
```

Before each trade, the number of contracts is

```math
n_j
=
\left\lfloor
\frac{
fW_j
}{
L
}
\right\rfloor
```

where $W_j$ is the available pre-trade capital and $f$ is the risk-budget fraction.

The risk-budget fraction controls position sizing. It is not a guaranteed maximum-loss percentage.

### Drawdown control

Running drawdown is measured from the highest capital level reached so far.

```math
DD_j
=
1-
\frac{
W_j
}{
\max_{k\le j}W_k
}
```

If the configured maximum drawdown limit is reached, the campaign stops immediately.

The maximum drawdown limit is the campaign's explicit risk kill switch.

If the sizing rule produces zero contracts, no new position is opened.

### Same-market counterfactuals

The campaign environment is generated before strategy execution.

Using the same integer campaign seed with another Research-qualified strategy recreates the same realized-volatility sequence and standardized Brownian shocks for corresponding trades.

This makes it possible to compare qualified strategies under the same exogenous market realization after the original selection decision has already been made.

These counterfactuals are diagnostic only and do not feed back into the original Research choice.

---

## Trade-level replay

Any executed campaign trade can be selected for detailed reconstruction.

The replay connects the realized market path with the mechanics of the hedge, including:

- the realized futures path,
- Black–76 delta evolution,
- hedge adjustments,
- accumulated transaction costs,
- futures hedge P&L,
- option payoff liability,
- terminal per-contract P&L,
- total effect on campaign capital.

The purpose is to understand how the path, hedge and transaction costs combined to produce the final result.

---

## Repository structure

```text
dynamic-delta-hedging/
│
├── main.ipynb
├── frictionless_convergence_validation.ipynb
│
└── dynamic_delta_hedging/
    ├── __init__.py
    ├── campaign.py
    ├── config.py
    ├── convergence.py
    ├── experiments.py
    ├── hedging_engine.py
    ├── market.py
    ├── metrics.py
    ├── plotting.py
    ├── presentation.py
    ├── pricing.py
    ├── quoting.py
    ├── research_analysis.py
    ├── robustness.py
    ├── selection.py
    ├── simulation.py
    ├── statistical_analysis.py
    ├── strategy_registry.py
    │
    └── strategies/
        ├── __init__.py
        ├── asset_tolerance.py
        ├── delta_tolerance.py
        ├── fixed_band.py
        ├── fixed_interval.py
        └── ww_inspired.py
```

[`main.ipynb`](main.ipynb) contains the complete Research-to-campaign workflow.

[`frictionless_convergence_validation.ipynb`](frictionless_convergence_validation.ipynb) contains the independent numerical convergence check.

The `dynamic_delta_hedging` package contains the pricing, simulation, hedging, strategy comparison, statistical analysis and campaign logic used by the notebooks.

---

## Running the project

The project uses the standard scientific Python stack:

- NumPy
- pandas
- SciPy
- Matplotlib
- Jupyter / IPython

The main workflow is contained in:

[`main.ipynb`](main.ipynb)

The convergence experiment can be run independently through:

[`frictionless_convergence_validation.ipynb`](frictionless_convergence_validation.ipynb)

Both notebooks are saved with the outputs of one representative execution.

Changing Research seeds, model inputs, Desk Mandate limits or campaign controls can change the qualified strategies, frontier composition, final decision and realized campaign path.

The methodology is therefore kept separate from any single saved numerical result.

---

## Scope

This is a quantitative research framework for studying dynamic hedge-policy selection for an option on futures.

It is not intended to reproduce a complete production market-making desk.

The model deliberately focuses on the interaction between option risk, discrete dynamic hedging, execution costs, downside-risk constraints and finite capital.

Several real-market effects remain outside the current scope, including:

- order flow and fill probability,
- adverse selection,
- multi-option inventory and portfolio netting,
- implied-volatility surface dynamics,
- stochastic volatility and jumps,
- margin and funding constraints,
- latency and exchange microstructure,
- physical delivery and commodity-specific operational constraints.

These omissions are deliberate. The project isolates the hedge-policy problem rather than attempting to model every component of a real derivatives desk.

---

## References

**Black, F. (1976).**  
*The Pricing of Commodity Contracts.*  
Journal of Financial Economics, 3(1–2), 167–179.  
https://doi.org/10.1016/0304-405X(76)90024-6

**Whalley, A. E., & Wilmott, P. (1997).**  
*An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs.*  
Mathematical Finance, 7(3), 307–324.  
https://doi.org/10.1111/1467-9965.00034

**Acerbi, C., & Tasche, D. (2002).**  
*On the Coherence of Expected Shortfall.*  
Journal of Banking & Finance, 26(7), 1487–1503.  
https://doi.org/10.1016/S0378-4266(02)00283-2

**Efron, B. (1979).**  
*Bootstrap Methods: Another Look at the Jackknife.*  
The Annals of Statistics, 7(1), 1–26.  
https://doi.org/10.1214/aos/1176344552

**Glasserman, P. (2004).**  
*Monte Carlo Methods in Financial Engineering.*  
Springer.  
https://doi.org/10.1007/978-0-387-21617-1

# F2Z PCS — implementer specification (v1, direct mode)

This document is the seam between the F2Z paper and an implementation. It states only what an implementer cannot derive from the paper: fixed parameters, exact object layouts, the order of messages and challenges, and the checks that must fire before each challenge.

It deliberately does not contain: the security argument, the relation-algebra formulation of the reduction, the intuition for why grand products replace integer sums, or any byte-level encoding. For the first three, read the paper. For the last, read the wire-format document — nothing here depends on it.

**Keywords.** MUST, MUST NOT, SHOULD carry their RFC 2119 meaning. "Derived" means the verifier computes the value itself; a derived value MUST NOT be read from the proof.

**Precedence.** Paper for semantics, this document for the profile, wire-format document for bytes. Where they disagree, the paper wins on what is being proved and this document wins on how the parties sequence it.

---

## 1. Parameters

**Fields.**

$$R = \mathbb{F}_q,\quad q = 2^{100} - 15 \qquad \mathbb{K} = \mathbb{F}_2[X]/(X^{128} + X^7 + X^2 + X + 1) \cong \mathbb{F}_{2^{128}}$$

[albert: $q$ is not a fixed prime. It is random generally. It is of about 106 bits]

Linear claims are evaluated in $R$. Packing, grand products, and the commitment opening happen in $\mathbb{K}$. $(\beta_v)_{v<128}$ with $\beta_v = X^v \bmod (X^{128}+X^7+X^2+X+1)$ is the ordered $\mathbb{F}_2$-basis of $\mathbb{K}$; every packing and coordinate extraction below uses it.

**Maps.** $\pi_q : \mathbb{Z} \to \mathbb{F}_q$ and $\pi_2 : \mathbb{Z} \to \mathbb{F}_2$ are reduction. $\operatorname{can}_q : \mathbb{F}_q \to [0,q)$ returns the unique integer representative — it is what makes the bounded exponent lifts in §4 well defined.

**Multilinear notation.** For an array $X$ indexed by $\{0,1\}^a$ with entries in a field $\mathbb{F}$, its multilinear extension is $\widetilde{X} : \mathbb{F}^a \to \mathbb{F}$ — evaluating it at a point returns **one** field element, not a vector. The equality polynomial is

$$\operatorname{eq} : \{0,1\}^a \times \mathbb{F}^a \to \mathbb{F}, \qquad \operatorname{eq}(z, r) := \prod_{h<a} \bigl((1-z_h)(1-r_h) + z_h r_h\bigr)$$

taken in whichever field the surrounding claim lives in. Coordinate order is the tensor order fixed in §2.

**Shape.** All of $t, s, W, w, d, m, m_p$ and every index are nonnegative integers. Public: $t$, $s$, $W$. Derived: $w = \log_2 W$, $d = t + w$, $m = d + s$, $m_p = m - 7$.

$2^d$ is the number of grand-product factors per column and $2^s$ the number of columns — the paper's tensor split $k_1 = 2^d$, $k_2 = 2^s$ with $k_1 k_2 = 2^m$.

**Typing convention.** Every value below is annotated with the ring it lives in at its definition site. Three rings are in play and they are not interchangeable: $\mathbb{Z}$ for exponents and bit arithmetic, $\mathbb{F}_q$ for the scalar linear claim, $\mathbb{K}$ for everything from the grand products onward.

**Admissibility.** The verifier MUST reject before any proof work unless

$$W \in 2^{\mathbb{N}}, \qquad d \ge 7, \qquad 22 \le m \le 35, \qquad (2^d + 1)(q - 1) < 2^{128} - 1.$$

For $q = 2^{100} - 15$ the last condition is exactly $d \le 27$. §4 derives it.

**Generator.** $g \in \mathbb{K}^\times$ with $\operatorname{ord}(g) = 2^{128} - 1$, checked as $g^{(2^{128}-1)/p} \ne 1$ for every prime $p \mid 2^{128}-1$. Full order plus the admissibility bound is what makes the fold exponent unique.

---

## 2. Committed objects

The witness is a tensor $D : \{0,1\}^t \times \{0,1\}^s \to [0, 2^W)$ of bounded integer cells.  [albert: W is always 1. This simplifies a lot the notation]

**Bit tensor.** With $i = (b \ll w) \mid j$ for $0 \le j < W$:

$$B[c][(b \ll w) \mid j] := \operatorname{bit}_j(D(b,c)), \qquad \operatorname{INT}(D(b,c)) = \sum_{j<W} 2^j\, B[c][(b \ll w) \mid j] \ \in \mathbb{Z}$$

so $B$ is a $2^s \times 2^d$ array of bits, $2^m$ in total. The index order — $c$ major, then $b$, then bit position $j$ — is fixed here and is the same order used by every coefficient vector in this document. Getting it wrong produces a valid-looking proof that fails only at the final opening.

**Domains of $B$.** The entries are canonical bits, but three different views of them appear below and mixing them silently is a correctness bug. Each use takes exactly the stated one:

- $B[c,i] \in \{0,1\} \subset \mathbb{Z}$ — integer arithmetic: $\operatorname{INT}$ above, and the column fold $\mu_c$ of §4 and P3.
- $\pi_2(B[c,i]) \in \mathbb{F}_2$ — packing and commitment, immediately below.
- $B_{\mathbb{K}} := \iota(B)$ with $\iota := \iota_{2 \to \mathbb{K}} \circ \pi_2$ [*albert*: this map doesn't make sense] sending a bit to $0, 1 \in \mathbb{K}$ — the grand-product claim and every MLE of a bit object (P4, P6).

**Packing and commitment.** For $0 \le i_{\text{hi}} < 2^{d-7}$:

$$P[c \cdot 2^{d-7} + i_{\text{hi}}] := \sum_{v=0}^{127} \pi_2\!\left(B[c][(i_{\text{hi}} \ll 7) \mid v]\right) \beta_v \in \mathbb{K}, \qquad P \in \mathbb{K}^{2^{m_p}}$$

$$O := \operatorname{Enc}_{\mathcal{C}}(P) \in \mathbb{K}^{|\mathcal{C}|}, \qquad \rho := \operatorname{Root}(O) \in \{0,1\}^{256}$$

$O$ is a codeword over $\mathbb{K}$ of the code's block length; $\rho$ is a Merkle root — a 32-byte digest, not a field element, and never an operand in any equation below.

The seven low index bits select the 128 basis coefficients. $O$ and $\rho$ are fixed for the whole protocol: no later step re-commits, and every step that appears to change the claim changes only the public coefficient attached to this same oracle.

---

## 3. The application seam

F2Z is a subprotocol. The caller supplies one linear claim about the committed witness; F2Z consumes nothing else and verifies nothing upstream.

**Entry points.**

$$\mathsf{ProveF2Z}(\mathsf{pp},\ x_{\text{core}},\ B;\ \mathsf{tr}), \qquad \mathsf{VerifyF2Z}(\mathsf{pp},\ x_{\text{core}},\ \mathsf{com},\ \pi;\ \mathsf{tr})$$

$$x_{\text{core}} = \bigl((t,s,W),\ (\boldsymbol{w}, \boldsymbol{w}', y),\ g\bigr), \qquad \mathsf{com} = (\rho)$$


[albert: what is $\pi$? The verifier does not receive $W$]

$\mathsf{tr}$ arrives carrying the caller's events. F2Z appends and returns it; it does not initialize or finalize it.

**Preconditions.** The caller MUST establish, and V2 re-checks:

1. The claim is about the witness committed under $\rho$.
2. $\boldsymbol{w} \in \mathbb{Z}^{2^t}$ with $0 \le w_b < 2^{100}$; $\boldsymbol{w}' \in \mathbb{F}_q^{2^s}$; $y \in \mathbb{F}_q$.
3. The caller's coefficient vector $\boldsymbol{v}$ [albert: what is this? Shouldn't this be an input or something?] factors as $v_{c,b} = \pi_q(w_b)\, w'_c$ [albert: a product of a field element and a non field element? Not well-defined]. A general coefficient is outside this profile — see Q8. [albert: what is "a general coefficient"?]
4. The claim holds:

$$y = \sum_{c} w'_c\, \pi_q\!\left(\underbrace{\sum_b w_b \operatorname{INT}(D(b,c))}_{\in\, \mathbb{Z}}\right) \quad \text{in } \mathbb{F}_q$$

The bit coefficients [albert: It's not very good practice to use non-standard expressions without having defined and motivated before, for example "bit coefficient"] are **derived** by F2Z, never transmitted and never taken from the caller: [albert: a no-op because $W=1$]

$$u_{c,b,j} := \pi_q(w_b 2^j)\, w'_c \ \in \mathbb{F}_q, \qquad i = (b \ll w) \mid j$$

For the MLE instantiation — the only one in v1 — the weights come from an evaluation point $(r_1, r_2) \in \mathbb{F}_q^t \times \mathbb{F}_q^s$ as $w_b = \operatorname{can}_q(\operatorname{eq}(b, r_1))$ and $w'_c = \operatorname{eq}(c, r_2)$, with $y$ the claimed MLE value. $w_b$ is an integer representative; §4 depends on that.

> **Open — see Q8.** Nothing here proves that an arbitrary application claim has this separable shape. v1 requires the caller to supply it and rejects otherwise.

[albert: I think the input should have $v$ s input, then the splitting into the w's should be part of the protocol]

---

## 4. The exponent bound

Everything in this section is integer arithmetic in $\mathbb{Z}$.

The protocol proves one grand product per column, whose exponent is the integer fold

$$\gamma_i := \operatorname{can}_q\!\left(\pi_q(w_b 2^j)\right) \in [0, q), \qquad \mu_c := \sum_i \gamma_i\, B[c,i] \ \in \mathbb{Z}$$

with $i = (b \ll w) \mid j$ ranging over $2^d$ positions. Since $B[c,i] \in \{0,1\}$,

$$0 \le \mu_c \le 2^d (q - 1).$$

The verifier learns $\mu_c$ only through $g^{\mu_c}$ [albert: the verivier needs to either know $\mu_c$ or $\mu_c$ mod q  so it can check Item 4 in the preconditions list], so the accepted exponent is unique exactly when no other admissible integer is congruent to it modulo $\operatorname{ord}(g)$. Bounding the gap between a claimed and a true fold by $(2^d + 1)(q-1)$ gives the admissibility condition of §1:

$$(2^d + 1)(q - 1) < \operatorname{ord}(g) = 2^{128} - 1.$$

At $q = 2^{100} - 15$ this is $d \le 27$, leaving $s \ge m - 27$. The verifier re-checks the per-column bound in V3 rather than trusting the derivation.

**No chunking.** v1 requires the shape to satisfy the bound outright; there is one fold, one grand product, and one GKR invocation. Weights are never split into digits.

> **Out of profile.** When the caller's modulus is too large to satisfy the bound — an extension evaluation domain, or a prime field with $q \ge |\mathbb{K}| - 1$ [albert: $q \ge (|\mathbb{K}| - 1)/2^t$ (or $/2^d$)] — the paper's construction inserts a projection round: the verifier samples a prime $q'$ from a public set $\mathcal{P}$ with $\max \mathcal{P} < (|\mathbb{K}|-1)/2^d$ (and, for an extension domain, a point $\alpha$), maps the claim into $\mathbb{F}_{q'}$, and continues with $q \gets q'$. v1 fixes $R = \mathbb{F}_q$ with $q = 2^{100}-15 < |\mathbb{K}|-1$, so the round never triggers and is not specified here. See Q5.

[albert: we should probably implement this?]

---

## 5. Protocol

Six steps, in this order, each running exactly once. There is no loop: one fold, one grand product, one GKR invocation, one linear claim. [albert: is this paragraph providing any info? Whould anybody ever think some steps run several times and that the execution loops around steps?]

This section is the only normative statement of the sequence [albert: could we avoid these AI expressions that don't mean anything :D ? What is a "normative statement", why can a section be a "statement", and why are the previous sections not "normative statements"?]. Each step gives the prover's action, the message and its channel, and the verifier's obligations in the order they must fire.

### Step 1 — Commit and bind the statement

**P1.** Build the committed objects of §2 and publish $\mathsf{com} = (\rho)$ as public input, not as a proof field:

$$B[c][(b \ll w) \mid j] = \operatorname{bit}_j(D(b,c)), \quad P[c 2^{d-7} + i_{\text{hi}}] = \sum_{v<128} \pi_2\!\left(B[c][(i_{\text{hi}} \ll 7) \mid v]\right)\beta_v$$

$$O = \operatorname{Enc}_{\mathcal{C}}(P), \qquad \rho = \operatorname{Root}(O)$$

**V1.**

1. Read $(t, s, W, g)$ from the statement; derive $w = \log_2 W$, $d = t+w$, $m = d+s$, $m_p = m-7$.
2. Require $W \in 2^{\mathbb{N}}$, $d \ge 7$, $22 \le m \le 35$, and $(2^d + 1)(q-1) < 2^{128}-1$.
3. Require $\operatorname{ord}(g) = 2^{128}-1$, checked as $g^{(2^{128}-1)/p} \ne 1$ for every prime $p \mid 2^{128}-1$.

**Transcript.** Absorb $\langle \text{domain} \rangle = (\text{magic}, \text{version})$, then $\langle \text{statement} \rangle = (d_{\text{src}}, t, s, W, g, \rho)$, where $d_{\text{src}}$ is the digest of the canonical source-statement bytes. Both MUST land before any proof-dependent squeeze.

### Step 2 — Bitify

**P2.** No message. Retain $B$ and its matrix form.

**V2.** Re-check the §3 preconditions on the arguments as received, and reject on failure:

$$|\boldsymbol{w}| = 2^t, \quad |\boldsymbol{w}'| = 2^s, \quad 0 \le w_b < 2^{100}, \quad w'_c, y \in \mathbb{F}_q$$

together with $x_{\text{core}}$ carrying the same $\rho$ as $\mathsf{com}$. Then derive, with no message and no challenge:

$$u_{c,b,j} = \pi_q(w_b 2^j)\, w'_c \ \in \mathbb{F}_q, \qquad \gamma_i = \operatorname{can}_q\!\left(\pi_q(w_b 2^j)\right) \in [0, q) \subset \mathbb{Z}, \qquad i = (b \ll w) \mid j$$

$\boldsymbol{\gamma} \in [0,q)^{2^d}$ is the integer weight vector the grand product exponentiates; $\boldsymbol{w}'$ stays in $\mathbb{F}_q$ and is applied only in the V3 reconstruction.

**Transcript.** Absorb $\langle \text{core seam} \rangle = (\boldsymbol{w}, \boldsymbol{w}', y)$ — after the derivation above, before the fold message.

This is re-typing by semilinearity, not a Booleanity round: $O$ already commits to the bits over $\mathbb{F}_2$, so no round proving $b \in \{0,1\}$ is needed or permitted.

### Step 3 — Fold

**P3.** With $i = (b \ll w) \mid j$, over $\mathbb{Z}$ with $B[c,i] \in \{0,1\}$:

$$\mu_c = \sum_i \gamma_i\, B[c,i] \in \mathbb{Z}, \qquad \nu_c = g^{\mu_c} \in \mathbb{K}^\times$$

Send $\boldsymbol{\nu} \in \mathbb{K}^{2^s}$ — **bound**. Then $\boldsymbol{\mu} \in \mathbb{Z}^{2^s}$ — **checked**, since $\rho$ and $\boldsymbol{\nu}$ already fix it.

**V3**, in this order:

1. Absorb $\boldsymbol{\nu}$.
2. Require $|\boldsymbol{\mu}| = 2^s$ and, for every $c$:

    $$0 \le \mu_c \le 2^d (q-1), \qquad g^{\mu_c} = \nu_c$$

    Both are needed: full order of $g$ plus the range bound is what makes the accepted exponent unique.

3. Require the reconstruction, which ties the folds back to the caller's claim:

    $$\sum_c w'_c\, \pi_q(\mu_c) = y \quad \text{in } \mathbb{F}_q$$

    Until this passes, the folds say nothing about $y$.

4. Derive $y_i = g^{\gamma_i} \in \mathbb{K}^\times$ for $0 \le i < 2^d$ — integer exponent, $\mathbb{K}$ result.
5. Squeeze the point $\zeta \in \mathbb{K}^s$ and derive the batched output claim

    $$e_0 = \widetilde{\boldsymbol{\nu}}(\zeta) \ \in \mathbb{K}$$

    where $\widetilde{\boldsymbol{\nu}}$ is the multilinear extension of the array $c \mapsto \nu_c$ over $\{0,1\}^s$.

**Transcript.** Absorb $\boldsymbol{\nu}$. Never absorb $\boldsymbol{\mu}$. Squeeze $\zeta$ as $s$ scalar squeezes, and only after checks 2 and 3 have passed.

### Step 4 — GKR reduction

**P4.** $\pi_{\text{red}} \leftarrow \mathsf{GKR.Prove}(\mathsf{pp}_{\text{GKR}}, \mathcal{I}_{\text{GKR}}, B; \mathsf{tr})$ with the input bundle

$$\mathcal{I}_{\text{GKR}} = \left(x_{\text{core}},\ \rho,\ \boldsymbol{y},\ \boldsymbol{\nu},\ \zeta,\ e_0\right)$$

**V4.** $(\mathsf{tr}', pt, \mu') \leftarrow \mathsf{GKR.Verify}(\mathsf{pp}_{\text{GKR}}, \mathcal{I}_{\text{GKR}}, \pi_{\text{red}}; \mathsf{tr})$. Require success and complete consumption of the fragment, then set $\mathsf{tr} \leftarrow \mathsf{tr}'$.

The claim GKR discharges, in $\mathbb{K}$ over the embedded bits:

$$\prod_i \left(1 + (y_i-1)\, B_{\mathbb{K}}[c,i]\right) = \nu_c \qquad (\forall c)$$

**Output contract.** GKR MUST return $pt \in \mathbb{K}^m$ and $\mu' \in \mathbb{K}$ with an MLE-shaped coefficient, against the same root:

$$a[z] = \operatorname{eq}(z, pt) \in \mathbb{K} \quad (z \in \{0,1\}^m), \qquad \widetilde{B_{\mathbb{K}}}(pt) = \mu' \in \mathbb{K}$$

A GKR instantiation returning a general coefficient does not satisfy this profile.

**Transcript.** GKR owns its own events. One constraint the PCS level imposes: the phase-$c$ claimed sum MUST be absorbed after the phase-link check and before the first phase-$c$ round message — see Q7.

### Step 5 — Ring switch

**P5.** Split $pt \in \mathbb{K}^m$ as $(r_{\text{lo}}, r_{\text{hi}})$ with $r_{\text{lo}} \in \mathbb{K}^7$ and $r_{\text{hi}} \in \mathbb{K}^{m_p}$, matching the $2^{m_p}$ entries of a bit plane. For $v < 128$:

$$B_v[c, i_{\text{hi}}] := B[c][(i_{\text{hi}} \ll 7) \mid v], \qquad s_v = \widetilde{\iota(B_v)}(r_{\text{hi}}) \in \mathbb{K}$$

Send $\boldsymbol{s} \in \mathbb{K}^{128}$ — **bound**.

**V5.** Absorb $\boldsymbol{s}$, then require the source gate before any squeeze:

$$\sum_{v=0}^{127} \operatorname{eq}(v, r_{\text{lo}})\, s_v = \mu' \quad \text{in } \mathbb{K}$$

Then squeeze $r'' \in \mathbb{K}^7$ and derive, with $[z]_u$ the $u$-th coordinate in the fixed $\mathbb{K}/\mathbb{F}_2$ basis:

$$\bar{s}_u = \sum_v [s_v]_u \beta_v \ \in \mathbb{K} \quad (u < 128), \qquad \Phi_{r''} : \mathbb{K} \to \mathbb{K}, \quad \Phi_{r''}(z) = \sum_u \operatorname{eq}(u, r'')[z]_u$$

$$\beta = \sum_u \operatorname{eq}(u, r'')\, \bar{s}_u \ \in \mathbb{K}, \qquad B_{\text{coef}} : \{0,1\}^{m_p} \to \mathbb{K}, \quad B_{\text{coef}}(Y) = \Phi_{r''}\!\left(\operatorname{eq}(r_{\text{hi}}, Y)\right)$$

$B_{\text{coef}}$ is evaluated on demand and MUST NOT be materialized as a $2^{m_p}$-entry vector. The gate uses no division and has no exceptional branch. Both parties derive $(B_{\text{coef}}, \beta)$ independently; neither is transmitted.

**Transcript.** Absorb $\boldsymbol{s}$, then squeeze $r''$ as 7 scalar squeezes.

### Step 6 — Open

**P6.** $\pi_{\text{open}} \leftarrow \mathsf{Open.Prove}(\rho, B_{\text{coef}}, \beta)$. Send on the opening protocol's own channels.

**V6.**

1. Require $\operatorname{dom}(B_{\text{coef}}) = \{0,1\}^{m_p}$ and that $B_{\text{coef}}$ equals the coefficient derived in V5 — it is never serialized.
2. Require $\mathsf{RecVerify}(\mathsf{pp}, \rho, B_{\text{coef}}, \beta; \mathsf{tr}) = 1$.

**Transcript.** Absorb $\langle \text{opening target} \rangle = (m_p, \beta)$ before invoking the opening verifier, which then owns its own events.

**Acceptance.** Return true only if every check above passed, no checked value is pending, and both proof streams are exactly exhausted.

---

## 6. Opening parameters

v1 instantiates the opening IOP as recursive Ligerito over $\mathbb{K}$. The paper's construction names WHIR with ring-switching at the Johnson bound instead; its experiments use Flock's Ligerito. See Q3 — the parameters below are Ligerito's and do not transfer.

Fixed for $22 \le m \le 35$. Every quantity in this section is a nonnegative integer in $\mathbb{Z}$ — level indices, exponents, counts, and difficulties alike. Implementations MUST use these integers directly and MUST NOT recompute them in floating point.

$$h_i = m - 11 - 3i, \qquad J_m = \min\{J \ge 1 : h_{J-1} \in \{3,4,5\}\}$$

$$r_i = 1 + i, \qquad k_i = \begin{cases} 4 & i = 0 \\ 3 & i > 0\end{cases}, \qquad Q_i = (183, 90, 60, 45, 36, 31, 27, 24)_i, \qquad o_i = \begin{cases} 0 & i = 0 \\ 1 & i > 0\end{cases}$$

$h_i$ is the message-column exponent, $r_i$ the inverse-rate exponent, $k_i$ the fold count, $Q_i$ the query count, $o_i$ the out-of-domain sample count. $Q_i$ and the grinding difficulties below carry the proximity security; changing either changes the security level.

**Base fold difficulties $f_i$.** Only the first $J_m$ entries of each row exist.

| $m$ | $(f_0, \ldots, f_{J_m-1})$ | $m$ | $(f_0, \ldots, f_{J_m-1})$  |
| --- | -------------------------- | --- | --------------------------- |
| 22  | 9, 6, 3                    | 29  | 16, 13, 10, 7, 5, 3         |
| 23  | 10, 7, 4, 1                | 30  | 17, 14, 11, 8, 6, 4         |
| 24  | 11, 8, 5, 2                | 31  | 18, 15, 12, 9, 7, 5         |
| 25  | 12, 9, 6, 3                | 32  | 19, 16, 13, 10, 8, 6, 3     |
| 26  | 13, 10, 7, 4, 2            | 33  | 20, 17, 14, 11, 9, 7, 4     |
| 27  | 14, 11, 8, 5, 3            | 34  | 21, 18, 15, 12, 10, 8, 5    |
| 28  | 15, 12, 9, 6, 4            | 35  | 22, 19, 16, 13, 11, 9, 6, 6 |

**Grinding.** Query difficulty is 16 at every level. Fold difficulty at level $k$, round $i$ is $\max(f_k - i, 0)$; difficulty zero means no seed, no nonce, and no predicate — not a nonce that trivially passes. Total nonce count is therefore $J_m + \sum_{k<J_m} \min(k_k, f_k)$.

**Query positions.** Each draw yields $r \in \mathbb{K}$ and proposes $q = \operatorname{low}_{64}(r) \bmod 2^{h_k + r_k}$, where $\operatorname{low}_{64}$ is the low 64-bit word of the element. (The source spec writes this as $r_{\text{lo}}$; it is unrelated to the ring-switch challenge split $r_{\text{lo}}$ in P5.) Repeat, discarding duplicates and counting every attempt, until $Q_k$ distinct positions exist, then sort ascending. The attempt count is transcript-derived, so both parties reach the same set.

---

## 7. Open questions for the paper author

Every item below is checked against `f2z-pcs` at `a860545`; line numbers are `paper/main.tex`. None should be closed by an implementer picking a behaviour.

**Q1 — Unresolved merge conflicts.** Eight conflict markers at lines 108, 208, 678, 729, 777, 813, 830, 2017. Five sit inside `c:core_iop` and `t:thm_core_IOPP`, and the sides differ in substance: `e:iopp_mle_claims` appears as $k_2$ copies of $\mathcal{R}_{\text{Lin}}$ on one side and a single $\mathsf{LIN}$ on the other. This spec reads the HEAD side. Needed: which side is current.

**Q2 — $k_1$ exponent.** `s:instantiation` grounds its analysis with "$k \le 2^{35}$ … and $k_1 \le 2^{14}$ following our rule of thumb", but the rule is $\log k_1 \approx 0.6 \log k$, restated at 1272 as $k_1 \approx k^{0.6}$, $k_2 = k^{0.4}$. $0.6 \times 35 = 21$; $2^{14}$ is what the $0.4$ factor gives. At $k_1 = 2^{21}$ the ceiling $\max\mathcal{P} < (|\mathbb{K}|-1)/k_1$ is $2^{107}$, not $2^{114}$, placing every minimum in Table 1 ($a^* = 108$–$112$) above it. Needed: which exponent, and whether the table needs recomputing.

**Q3 — Opening protocol identity.** `c:core_iop` (660) and `s:instantiation` (1122) name WHIR with ring-switching at the Johnson bound; the experiments (234) use Flock's Ligerito, and 2027 pins an RS code of rate $1/8$. §6's parameter tables are Ligerito's. Needed: which is normative for the benchmark.

**Q4 — Chunking removal.** Earlier drafts of this spec split each $w_b$ into $c_w$-bit digits with one grand product per active digit. Nothing in the current paper does this; §4 now follows the direct bound with prime projection as the escape. Needed: confirmation that chunking is not pending re-add.

**Q5 — The exact integer fold.** The note at 729 states that after the Step-3 $q'$ update the claim $\prod_i g^{\gamma_i f_{ij}} = g^{\mu_j}$ is unsatisfiable for an honest prover, and that the prover must send $\mu_j' = \langle f_{*j}, \gamma\rangle$ with the verifier checking $\mu_j' \equiv \mu_j(\alpha) \pmod{q'}$. Needed: whether that enters the construction text, and whether leaving the whole projection path unspecified is acceptable for a profile with $q < |\mathbb{K}|-1$, where it never triggers.

**Q6 — Which tensor factor carries the powers of two.** `r:tensor_dec_u` suggests $u^{(1)} := v^{(1)}$ and $u^{(2)} := v^{(2)} \otimes (1, \pi(2), \ldots, \pi(2^{B-1})) \otimes (\psi(\gamma_1), \ldots)$, putting the $2^k$ powers on the $u^{(2)}$ side. §2 does the opposite: the per-cell weight and the bit-position powers $2^j$ both land in $u^{(1)}$, giving $k_1 = 2^d$ and the admissibility bound $d \le 27$. The choice moves $k_1$, so it moves the admissible shape. Needed: whether the remark's split is intended or illustrative. (`s:parameter_instantiation`, cited there for optimal split dimensions, has no `\label` in the source.)

**Q7 — Phase-$c$ claimed sum.** `c:gkr_grand_product` runs one sumcheck per layer over $(b,c)$ jointly and has the verifier derive $C^{(i+1)} = a_0(1-\beta) + a_1\beta$; nothing about it is transmitted. The merged-forest implementation splits each layer into two sumchecks, over $b$ then $c$, producing an intermediate claimed sum with no counterpart in the paper and no evident binding — when $\operatorname{eq}(r_x, z_x) = 0$ the phase link does not determine it, so it appears to need absorbing rather than deriving. The margin reply at 1322 holds the two equivalent and suggests changing the implementation. Needed: implementation collapses to one sumcheck, or the paper covers the split and pins the binding rule.

**Q8 — Non-separable claims above $m = 27$.** `c:core_iop` admits the trivial decomposition ($k_1 = k$, $k_2 = 1$). Under it the exponent bound gives $m \le 27$ at $q = 2^{100}-15$, below the $m = 35$ target, so a nontrivial split is a feasibility requirement rather than a proof-size optimization — and it needs the caller's coefficient to factor as $v_{c,b} = \pi_q(w_b) w'_c$ (§3). Needed: whether a non-separable claim is out of scope at $m > 27$.

**Q9 — Composed soundness.** $\kappa_0 = \deg(R)/q_{\min} + (\log(k_1+1) + \log(q-1))/(|\mathcal{P}| \lfloor \log q_{\min}\rfloor)$, which is $0$ for a prime $R$ with the projection round untriggered. Still needed: that the concrete GKR edge realizes the abstract reduction, and that ring-switch and opening compose to the abstract opening edge, with the stated loss.

**Minor.** 27 unresolved `\cref{?}`. Two cited labels do not exist: `s:proof_size_opt` (Round 1, the variant committing to $\mu$ rather than sending it) and `s:parameter_instantiation`.

## Deliberately excluded

For reviewers checking that nothing load-bearing was dropped:

- **Relation-algebra formulation.** The chain and its edge notation are in the paper; an implementation never materializes a relation.
- **Virtual mode.** The general-coefficient branch, its adjoint rewrite, and its LinCheck. v1 is direct-only and rejects virtual inputs; specifying a mode nothing implements adds surface without value.
- **Byte encodings.** Container, framing, canonicality, Merkle serialization, parser conformance. Separate document — no cryptographic decision in this one depends on them.
- **Grand-product motivation.** Why bounded integer sums become $\mathbb{K}$ products. Paper.
- **Weight chunking.** Earlier drafts split each $w_b$ into $c_w$-bit digits and ran one grand product per active digit. The paper does not do this: it bounds the single fold directly (§4) and, when that bound cannot be met, projects onto a sampled prime instead (Q4). Removed, along with the active set, the per-chunk loop, and the cross-chunk batching step.

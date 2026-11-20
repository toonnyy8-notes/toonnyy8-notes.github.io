SCORE 12/12

LO1 correct. First-token weight, softmax sum=1, value near 0 all present.
LO2 correct. exp(sink) in the denominator, learnable, no value (Eq. 1); G=sigma(xW_theta) per head (Eq. 2). The tester did not give the W_theta dimension d_model x 1, but the dimension is not required in the reference question.
LO3 correct. Value norm near 0; 70.38 vs 70.36; All = 64.10. page.txt line 145 states them.
LO4 correct. O=(1-A_sink)·sum(A~ v), A~=A/(1-A_sink), G=1-A_sink; Eq. 16 and Eq. 23 forms; Vanilla error A_0 v_0; Sink exact.
LO5 correct. Heads are experts, the sink gate is the router; the two shared designs are head-specific and sigmoid form.
LO6 correct. Query clusters near k_0, key pushed away, dot-product differences compressed, softmax precision drops, long-context noise.
LO7 correct. Imp = mean of G over tokens (Eq. 5); CV = std/mean averaged over layers (Eq. 6); 0.3 / 0.3464 / 1.1547 with population std.
LO8 correct. Eq. 7, lambda = 1e-4, N·CV^2 = 5.333.
LO9 correct. Head pinning; top-m shared heads excluded, (N_h-m)[CV]^2 over routed heads; lambda = 1e-2.
LO10 correct. Fused kernels do not output weights, eager is too slow; kernel LSE used; Sink and Vanilla formulas; under 2%; LSE_{¬0} not obtainable from the kernel, cancellation as A_0→1.
LO11 correct. All Table 1 and Table 2 numbers match.
LO12 correct. Chain Value Drain → Attention Sink → Implicit Gating → Head Collapse; the Gated gate is explicit from the start, so collapse is direct.

Page contradictions with the reference: none found. The page is consistent with the reference and adds the 6.26 note, the 14B numbers and the expm1 advice.

Page fixes: none needed.

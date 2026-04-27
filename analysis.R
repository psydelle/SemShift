# analysis.R
# All statistical models for the SemShift project.
# Run from project root: Rscript analysis.R
# Reads:  output/analysis_dataset.csv  (item-level, from notebook cell 5/export)
#         data/experiment_data_anonymised.csv  (trial-level)
#         output/verb_noun_deltas.csv          (delta + LSCD metrics)
# Prints: all model summaries to stdout

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)   # adds p-values to lmer() via Satterthwaite df
  library(effectsize) # cohens_d()
})

POOL_MIN <- 50

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

item  <- read_csv("output/analysis_dataset.csv",  show_col_types = FALSE)
exp   <- read_csv("data/experiment_data_anonymised.csv", show_col_types = FALSE)
deltas <- read_csv("output/verb_noun_deltas.csv", show_col_types = FALSE)

cat("Item-level data:", nrow(item), "rows\n")
cat("Trial-level data:", nrow(exp), "rows\n")

# ---------------------------------------------------------------------------
# RQ1 — Paired t-test: verb_delta_cos vs noun_delta_cos
# ---------------------------------------------------------------------------

cat("\n\n====== RQ1: Paired t-test (verb vs noun delta cosine) ======\n")

t1 <- t.test(item$verb_delta_cos, item$noun_delta_cos, paired = TRUE)
print(t1)

diff <- item$verb_delta_cos - item$noun_delta_cos
d1   <- cohens_d(diff)
cat("Cohen's d:", round(d1$Cohens_d, 3), "\n")
cat("Interpretation: negative d => verb cos < noun cos => verbs MORE mutable (predicted)\n")

cat("\n--- By condition ---\n")
for (cond in unique(item$condition)) {
  sub  <- item %>% filter(condition == cond)
  tc   <- t.test(sub$verb_delta_cos, sub$noun_delta_cos, paired = TRUE)
  dc   <- cohens_d(sub$verb_delta_cos - sub$noun_delta_cos)
  cat(sprintf("%14s  n=%d  t=%.3f  p=%.4f  d=%.3f\n",
              cond, nrow(sub), tc$statistic, tc$p.value, dc$Cohens_d))
}

# ---------------------------------------------------------------------------
# RQ1b — Mixed model: word_type * condition + (1|item)
# ---------------------------------------------------------------------------

cat("\n\n====== RQ1b: OLS interaction (word_type x condition) ======\n")
# Note: lmer(... + (1|item)) produces a singular fit here because each item
# has exactly 2 rows (verb, noun) and word_type perfectly partitions them —
# no residual between-item variance remains. OLS is equivalent and correct.

long <- bind_rows(
  item %>% select(item, condition, verb_delta_cos, verb_pool_size) %>%
    rename(delta_cos = verb_delta_cos, pool_size = verb_pool_size) %>%
    mutate(word_type = "verb"),
  item %>% select(item, condition, noun_delta_cos, noun_pool_size) %>%
    rename(delta_cos = noun_delta_cos, pool_size = noun_pool_size) %>%
    mutate(word_type = "noun")
) %>% drop_na(delta_cos) %>%
  mutate(
    word_type = relevel(factor(word_type), ref = "noun"),
    condition = relevel(factor(condition), ref = "Productive")
  )

cat("Long format:", nrow(long), "rows\n\n")

cat("--- M_base (no covariates) ---\n")
m_base <- lm(delta_cos ~ word_type * condition, data = long)
print(summary(m_base))

cat("\n--- M_pool (+ pool_size) ---\n")
m_pool <- lm(delta_cos ~ word_type * condition + pool_size, data = long)
print(summary(m_pool))

# Pool ≥ 50 robustness
cat(sprintf("\n--- RQ1b replicated (pool >= %d) ---\n", POOL_MIN))
long50 <- long %>% filter(pool_size >= POOL_MIN)
cat("Items:", long50 %>% distinct(item) %>% nrow(), "\n\n")
m50 <- lm(delta_cos ~ word_type * condition, data = long50)
print(summary(m50))

# ---------------------------------------------------------------------------
# RQ2 — Item-level OLS: does delta predict RT?  (known null, brief)
# ---------------------------------------------------------------------------

cat("\n\n====== RQ2: Item-level OLS (delta ~ iRT_AJT) ======\n")

for (formula in c("iRT_AJT ~ verb_delta_cos",
                  "iRT_AJT ~ noun_delta_cos",
                  "iRT_AJT ~ verb_delta_cos * condition",
                  "iRT_AJT ~ noun_delta_cos * condition")) {
  fit <- lm(as.formula(formula), data = item)
  cat(sprintf("\n--- %s ---\n", formula))
  cat(sprintf("  R2=%.4f  adj-R2=%.4f  AIC=%.1f\n",
              summary(fit)$r.squared,
              summary(fit)$adj.r.squared,
              AIC(fit)))
  print(coef(summary(fit)))
}

# ---------------------------------------------------------------------------
# RQ2b — Trial-level mixed models: log(RT) ~ delta_z * condition + (1|participant)
# ---------------------------------------------------------------------------

cat("\n\n====== RQ2b: Trial-level mixed models ======\n")

# Merge deltas; use lowercase verb/noun columns already present in exp
trials <- exp %>%
  left_join(
    deltas %>% select(verb, noun, verb_delta_cos, noun_delta_cos,
                      verb_pool_size, noun_pool_size),
    by = c("verb", "noun")
  )

# Filter: accepted trials, RT >= 200ms, 3-SD per-participant ceiling
t_data <- trials %>%
  filter(!is.na(verb_delta_cos), Response_AJT == "y", RT_AJT >= 200) %>%
  group_by(participant_id) %>%
  mutate(rt_mean = mean(RT_AJT), rt_sd = sd(RT_AJT)) %>%
  filter(RT_AJT <= rt_mean + 3 * rt_sd) %>%
  ungroup() %>%
  mutate(
    log_RT       = log(RT_AJT),
    verb_delta_z = as.numeric(scale(verb_delta_cos)),
    noun_delta_z = as.numeric(scale(noun_delta_cos)),
    Condition    = relevel(factor(Condition), ref = "Productive")
  )

cat("Trials (accepted, trimmed):", nrow(t_data), "\n")
cat("Participants:", n_distinct(t_data$participant_id), "\n")
cat("Items:", n_distinct(t_data$Item), "\n\n")

cat("--- M_verb: log_RT ~ verb_delta_z * Condition + (1|participant) ---\n")
m_verb <- lmer(log_RT ~ verb_delta_z * Condition + (1 | participant_id),
               data = t_data, REML = FALSE)
print(summary(m_verb))

cat("\n--- M_noun: log_RT ~ noun_delta_z * Condition + (1|participant) ---\n")
m_noun <- lmer(log_RT ~ noun_delta_z * Condition + (1 | participant_id),
               data = t_data, REML = FALSE)
print(summary(m_noun))

# Pool >= 50 robustness
cat(sprintf("\n--- RQ2b replicated (pool >= %d, n=%d trials) ---\n",
            POOL_MIN,
            nrow(t_data %>% filter(verb_pool_size >= POOL_MIN, noun_pool_size >= POOL_MIN))))

t50 <- t_data %>% filter(verb_pool_size >= POOL_MIN, noun_pool_size >= POOL_MIN)

m_verb50 <- lmer(log_RT ~ verb_delta_z * Condition + (1 | participant_id),
                 data = t50, REML = FALSE)
print(summary(m_verb50))

m_noun50 <- lmer(log_RT ~ noun_delta_z * Condition + (1 | participant_id),
                 data = t50, REML = FALSE)
print(summary(m_noun50))

# ---------------------------------------------------------------------------
# RQ1c — Three-condition comparison: Productive vs Collocation vs Idiom
# ---------------------------------------------------------------------------

cat("\n\n====== RQ1c: Three-condition comparison (Productive / Collocation / Idiom) ======\n")

idiom_deltas <- read_csv("output/verb_noun_deltas_idioms.csv", show_col_types = FALSE)
cat("Idiom items:", nrow(idiom_deltas), "\n")

# Paired t-test within idiom condition
cat("\n--- Idiom: paired t-test (verb vs noun delta cosine) ---\n")
idiom_c <- idiom_deltas %>% drop_na(verb_delta_cos, noun_delta_cos)
t_idiom  <- t.test(idiom_c$verb_delta_cos, idiom_c$noun_delta_cos, paired = TRUE)
print(t_idiom)
d_idiom <- cohens_d(idiom_c$verb_delta_cos - idiom_c$noun_delta_cos)
cat("Cohen's d:", round(d_idiom$Cohens_d, 3), "\n")

# Mean delta by condition across all three
cat("\n--- Mean delta cosine by condition ---\n")
all3 <- bind_rows(
  item        %>% select(item, condition, verb_delta_cos, noun_delta_cos, iRT_AJT),
  idiom_deltas %>% select(item, condition, verb_delta_cos, noun_delta_cos, iRT_AJT)
)
all3 %>%
  group_by(condition) %>%
  summarise(n = n(),
            verb_mean = mean(verb_delta_cos, na.rm = TRUE),
            noun_mean = mean(noun_delta_cos, na.rm = TRUE),
            diff      = verb_mean - noun_mean,
            .groups = "drop") %>%
  print()

# OLS: word_type * condition across all three conditions
cat("\n--- M_3cond: delta_cos ~ word_type * condition (all three conditions) ---\n")
long3 <- bind_rows(
  all3 %>% select(item, condition, verb_delta_cos) %>%
    rename(delta_cos = verb_delta_cos) %>% mutate(word_type = "verb"),
  all3 %>% select(item, condition, noun_delta_cos) %>%
    rename(delta_cos = noun_delta_cos) %>% mutate(word_type = "noun")
) %>% drop_na(delta_cos) %>%
  mutate(
    word_type = relevel(factor(word_type), ref = "noun"),
    condition = relevel(factor(condition), ref = "Productive")
  )

m_3cond <- lm(delta_cos ~ word_type * condition, data = long3)
print(summary(m_3cond))

# RQ2 for idioms: does delta predict RT?
cat("\n--- Idiom RQ2: does delta predict iRT? ---\n")
for (f in c("iRT_AJT ~ verb_delta_cos", "iRT_AJT ~ noun_delta_cos")) {
  fit <- lm(as.formula(f), data = idiom_c)
  cat(sprintf("\n%s:  R2=%.4f  adj-R2=%.4f\n", f,
              summary(fit)$r.squared, summary(fit)$adj.r.squared))
  print(coef(summary(fit)))
}

# ---------------------------------------------------------------------------
# RQ1d — APD and clustering: can embeddings distinguish idiom from productive?
#
# If the model captures idiomaticity, idiom KWICs should be more variable
# (some sentences literal, some idiomatic) → higher APD, more clusters.
# If APD(idiom) ≈ APD(productive), the metric is likely blind to idiomaticity.
# ---------------------------------------------------------------------------

cat("\n\n====== RQ1d: APD and clustering — idiom vs productive vs collocation ======\n")

apd_data <- bind_rows(
  deltas       %>% select(condition, verb_apd, noun_apd, verb_n_clusters, noun_n_clusters),
  idiom_deltas %>% select(condition, verb_apd, noun_apd, verb_n_clusters, noun_n_clusters)
) %>% mutate(condition = factor(condition, levels = c("Productive", "Idiom", "Collocation")))

cat("\n--- Mean APD and cluster count by condition ---\n")
apd_data %>%
  group_by(condition) %>%
  summarise(
    n              = n(),
    verb_apd       = mean(verb_apd,        na.rm = TRUE),
    noun_apd       = mean(noun_apd,        na.rm = TRUE),
    verb_clusters  = mean(verb_n_clusters, na.rm = TRUE),
    noun_clusters  = mean(noun_n_clusters, na.rm = TRUE),
    .groups = "drop"
  ) %>% print()

cat("\n--- t-tests: Idiom vs Productive ---\n")
for (col in c("verb_apd", "noun_apd", "verb_n_clusters", "noun_n_clusters")) {
  idiom_vals <- apd_data %>% filter(condition == "Idiom")    %>% pull(!!col) %>% na.omit()
  prod_vals  <- apd_data %>% filter(condition == "Productive") %>% pull(!!col) %>% na.omit()
  tt <- t.test(idiom_vals, prod_vals)
  cat(sprintf("  %-20s  idiom=%.3f  prod=%.3f  t=%.3f  p=%.4f\n",
              col, mean(idiom_vals), mean(prod_vals), tt$statistic, tt$p.value))
}

cat("\n--- t-tests: Idiom vs Collocation ---\n")
for (col in c("verb_apd", "noun_apd", "verb_n_clusters", "noun_n_clusters")) {
  idiom_vals  <- apd_data %>% filter(condition == "Idiom")       %>% pull(!!col) %>% na.omit()
  colloc_vals <- apd_data %>% filter(condition == "Collocation") %>% pull(!!col) %>% na.omit()
  tt <- t.test(idiom_vals, colloc_vals)
  cat(sprintf("  %-20s  idiom=%.3f  colloc=%.3f  t=%.3f  p=%.4f\n",
              col, mean(idiom_vals), mean(colloc_vals), tt$statistic, tt$p.value))
}

cat("\nDone.\n")

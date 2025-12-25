library(tidyverse)
library(lme4)
library(emmeans)

# 1) Read data
d <- readr::read_csv("ally.csv") %>%
  mutate(
    neet_year = factor(neet_year),
    model = factor(model),
    run = factor(run),
    question_code = factor(question_code),
    marks = as.integer(marks),
    option_selected = factor(option_selected),
    correct_option  = factor(correct_option)
  )

# (Optional but recommended) set reference model for "vs reference" tests
ref_model <- "GPT 5.2"
d <- d %>% mutate(model = relevel(model, ref = ref_model))

# 2) Fit GLMM (random intercept for question)
fit0 <- glmer(marks ~ neet_year + (1 | question_code),
              data = d, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))

fit1 <- glmer(marks ~ model + neet_year + (1 | question_code),
              data = d, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))

# 3) Global test: does "model" improve fit?
global_test <- anova(fit0, fit1, test = "Chisq")
global_test
# Report p-value from this LRT as the overall evidence of between-model differences.

# 4) Table of adjusted probabilities (like your GLMM table)
emm_prob <- emmeans(fit1, ~ model, type = "response")
emm_prob_df <- as.data.frame(emm_prob) %>%
  arrange(desc(prob))
emm_prob_df%>%write.csv('emm_prob_df.csv')

# 5A) Pairwise model comparisons (ALL pairs) with multiplicity correction
# Work on LINK scale (log-odds differences), then exponentiate to OR.
emm_link <- emmeans(fit1, ~ model)  # default: link (logit) scale

pairs_all <- pairs(emm_link, adjust = "tukey") %>%
  summary(infer = TRUE)

pairs_all_df <- as.data.frame(pairs_all) %>%
  mutate(
    OR = exp(estimate),
    OR_low = exp(asymp.LCL),
    OR_high = exp(asymp.UCL)
  ) %>%
  select(contrast, estimate, SE, df, z.ratio, p.value, OR, OR_low, OR_high) %>%
  arrange(p.value)

pairs_all_df%>%write.csv('pairs_all_df.csv')
# Interpretation:
# - estimate = log-odds difference (model1 - model2)
# - OR > 1 => model1 has higher odds of correct response than model2
# - p.value is Tukey-adjusted for multiple comparisons

# 5B) Comparisons vs the reference model only (often better for main manuscript)
vs_ref <- contrast(emm_link, method = "trt.vs.ctrl", ref = 1, adjust = "holm") %>%
  summary(infer = TRUE)

vs_ref_df <- as.data.frame(vs_ref) %>%
  mutate(
    OR = exp(estimate),
    OR_low = exp(asymp.LCL),
    OR_high = exp(asymp.UCL)
  ) %>%
  select(contrast, estimate, SE, z.ratio, p.value, OR, OR_low, OR_high) %>%
  arrange(p.value)

vs_ref_df %>%write.csv('vs_ref_df.csv')


# Run to Run Stability.

stab <- d %>%
  select(model, neet_year, question_code, run, option_selected) %>%
  pivot_wider(names_from=run, values_from=option_selected) %>%
  mutate(all_same = (`1` == `2`) & (`2` == `3`)) %>%
  group_by(model) %>%
  summarise(p_all_same = mean(all_same, na.rm=TRUE), .groups="drop") %>%
  arrange(desc(p_all_same))

stab

# Majority Vote accuracy per model.

maj <- d %>%
  group_by(model, neet_year, question_code) %>%
  summarise(n_correct = sum(marks), .groups="drop") %>%
  mutate(maj_correct = as.integer(n_correct >= 2))

maj %>%
  group_by(model) %>%
  summarise(maj_acc = mean(maj_correct), .groups="drop") %>%
  arrange(desc(maj_acc))
  
  
  # Difficulty Bins
  
  item_diff <- d %>%
  group_by(question_code) %>%
  summarise(p_correct = mean(marks), .groups="drop") %>%
  mutate(diff_bin = ntile(p_correct, 3)) %>%
  mutate(diff_bin = factor(diff_bin, labels=c("Hard","Medium","Easy")))

d2 <- d %>% left_join(item_diff, by="question_code")

d2 %>%
  group_by(model, diff_bin) %>%
  summarise(acc = mean(marks), .groups="drop") %>%
  ggplot(aes(diff_bin, acc, group=model)) +
  geom_line(alpha=0.4) +
  geom_point(alpha=0.6) +
  labs(x="Question difficulty (by overall p_correct)", y="Accuracy")
  
  # Option Choice Bias
  
  opt_bias <- d %>%
  filter(option_selected %in% c("A","B","C","D")) %>%
  count(model, option_selected) %>%
  group_by(model) %>%
  mutate(p = n/sum(n)) %>%
  ungroup()

opt_bias

# Forest plot

library(tidyverse)

emm <- read_csv("emm_prob_df.csv") %>%
  # emmeans typically names CIs as asymp.LCL / asymp.UCL on response scale
  rename(
    acc = prob,
    lo  = asymp.LCL,
    hi  = asymp.UCL
  ) %>%
  arrange(acc) %>%
  mutate(model = factor(model, levels = model))

ggplot(emm, aes(x = acc, y = model)) +
  geom_vline(xintercept = 0.25, linetype = "dashed", alpha = 0.4) + # optional reference
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2) +
  geom_point(size = 2) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  labs(x = "GLMM-adjusted probability of correct (95% CI)",
       y = NULL) +
  theme_bw()
  
  # Stability Plot
  
  library(tidyverse)

stab <- read_csv("stability.csv") %>%
  arrange(p_all_same) %>%
  mutate(model = factor(model, levels = model))

ggplot(stab, aes(x = p_all_same, y = model)) +
  geom_col() +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  labs(x = "Run-to-run stability (all 3 runs identical)", y = NULL) +
  theme_bw()
  
  # Accuracy versus stability Plot
  
  library(tidyverse)

emm <- read_csv("emm_prob_df.csv") %>%
  rename(acc = prob, lo = asymp.LCL, hi = asymp.UCL)

stab <- read_csv("stability.csv")

plot_df <- emm %>%
  inner_join(stab, by = "model")

ggplot(plot_df, aes(x = p_all_same, y = acc, label = model)) +
  geom_point(size = 2) +
  ggrepel::geom_text_repel(max.overlaps = 50, size = 3) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(x = "Stability (all 3 runs identical)",
       y = "GLMM-adjusted accuracy") +
  theme_bw()
  
  
  # Yearwise model Performance Plot
  
  library(tidyverse)
library(lme4)
library(emmeans)

d <- read_csv("ally.csv") %>%
  mutate(
    neet_year = factor(neet_year),
    model = factor(model),
    question_code = factor(question_code),
    marks = as.integer(marks)
  )

fit_int <- glmer(marks ~ model * neet_year + (1|question_code),
                 data = d, family = binomial,
                 control = glmerControl(optimizer="bobyqa"))

emm_y <- as.data.frame(emmeans(fit_int, ~ model | neet_year, type="response")) %>%
  rename(acc = prob, lo = asymp.LCL, hi = asymp.UCL)

# Plot only selected models (optional) to avoid clutter
# keep_models <- c("GPT 5.2", "Gemini 3 Flash", "Claude 3.5 Sonnet", "GPT-4o")
# emm_y <- emm_y %>% filter(model %in% keep_models)

ggplot(emm_y, aes(x = neet_year, y = acc, group = model)) +
  geom_line(alpha = 0.5) +
  geom_point(alpha = 0.7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.1, alpha = 0.25) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(x = "NEET-PG year", y = "Adjusted accuracy (95% CI)") +
  theme_bw()
  
  # Rerun Choice Plot
  
  library(tidyverse)

d <- read_csv("ally.csv") %>%
  mutate(
    neet_year = factor(neet_year),
    model = factor(model),
    question_code = factor(question_code),
    run = factor(run),
    marks = as.integer(marks)
  )

single <- d %>%
  group_by(model) %>%
  summarise(single_acc = mean(marks), .groups="drop")

maj <- d %>%
  group_by(model, neet_year, question_code) %>%
  summarise(n_correct = sum(marks), .groups="drop") %>%
  mutate(maj_correct = as.integer(n_correct >= 2)) %>%
  group_by(model) %>%
  summarise(maj_acc = mean(maj_correct), .groups="drop")

lift <- single %>%
  inner_join(maj, by="model") %>%
  mutate(lift = maj_acc - single_acc) %>%
  arrange(lift) %>%
  mutate(model = factor(model, levels = model))

ggplot(lift, aes(x = lift, y = model)) +
  geom_vline(xintercept = 0, linetype="dashed", alpha=0.5) +
  geom_col() +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(x = "Majority-vote lift over single-run accuracy", y = NULL) +
  theme_bw()
  
  
  # Option Choice Plot
  
  library(tidyverse)

opt <- read_csv("opt_bias.csv")  # columns: model, option_selected, n, p

# Order models by GLMM adjusted accuracy (optional)
emm <- read_csv("emm_prob_df.csv") %>% arrange(desc(prob))
opt <- opt %>%
  left_join(emm %>% select(model, prob), by="model") %>%
  mutate(model = fct_reorder(model, prob))

ggplot(opt, aes(x = model, y = p, fill = option_selected)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Proportion of selected options", fill = "Option") +
  theme_bw()
  
  library(tidyverse)

d <- read_csv("ally.csv") %>%
  mutate(
    model = factor(model),
    question_code = factor(question_code),
    marks = as.integer(marks)
  )

item_diff <- d %>%
  group_by(question_code) %>%
  summarise(p_correct = mean(marks), .groups="drop")

# Pick top 6 models to keep plot readable
top_models <- d %>%
  group_by(model) %>%
  summarise(acc = mean(marks), .groups="drop") %>%
  arrange(desc(acc)) %>%
  slice_head(n = 6) %>%
  pull(model)
  
  
  # Question Difficulty Plot

heat <- d %>%
  filter(model %in% top_models) %>%
  group_by(model, question_code) %>%
  summarise(acc_q = mean(marks), .groups="drop") %>%
  left_join(item_diff, by="question_code") %>%
  mutate(question_code = fct_reorder(question_code, p_correct))

ggplot(heat, aes(x = question_code, y = model, fill = acc_q)) +
  geom_tile() +
  coord_flip() +
  labs(x = "Questions (ordered by overall difficulty)", y = NULL, fill = "Accuracy") +
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
        
  # Forest plot by family
  
  
  

library(tidyverse)
library(lme4)
library(emmeans)

add_family <- function(x) {
  case_when(
    str_detect(x, regex("^GPT", ignore_case = TRUE)) ~ "OpenAI GPT",
    str_detect(x, regex("^Gemini", ignore_case = TRUE)) ~ "Google Gemini",
    str_detect(x, regex("^Claude", ignore_case = TRUE)) ~ "Anthropic Claude",
    str_detect(x, regex("^Llama", ignore_case = TRUE)) ~ "Meta Llama",
    str_detect(x, regex("^DeepSeek", ignore_case = TRUE)) ~ "DeepSeek",
    str_detect(x, regex("^Kimi", ignore_case = TRUE)) ~ "Kimi",
    TRUE ~ "Other"
  )
}

d <- readr::read_csv("ally.csv") %>%
  mutate(
    neet_year = factor(neet_year),
    model = factor(model),
    question_code = factor(question_code),
    marks = as.integer(marks)
  )

emm_y <- d %>%
  group_by(neet_year) %>%
  group_split() %>%
  purrr::map_dfr(function(df_year) {
    y <- unique(df_year$neet_year)

    fit_y <- glmer(
      marks ~ model + (1 | question_code),
      data = df_year, family = binomial,
      control = glmerControl(optimizer = "bobyqa")
    )

    as.data.frame(emmeans(fit_y, ~ model, type = "response")) %>%
      transmute(
        neet_year = y,
        model,
        acc = prob,
        lo = asymp.LCL,
        hi = asymp.UCL
      )
  }) %>%
  mutate(family = add_family(as.character(model)))

# Plot: year-wise, faceted by family
ggplot(emm_y, aes(x = neet_year, y = acc, group = model, color = model)) +
  geom_line(alpha = 0.6) +
  geom_point(alpha = 0.8) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.1, alpha = 0.25) +
  facet_wrap(~ family, scales = "free_y") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(x = "NEET-PG year", y = "Adjusted accuracy (95% CI)", color = "Model") +
  theme_bw()
  
  # Rasch Plot with automatic removal of 1 category item.
  
  library(tidyverse)
library(mirt)

# ---- read + build person-by-item matrix ----
d <- readr::read_csv("ally.csv") %>%
  mutate(
    question_code = as.character(question_code),
    marks = as.integer(marks),
    person_id = paste0(model, "::run", run)
  ) %>%
  select(person_id, model, run, question_code, marks) %>%
  distinct()

resp_wide <- d %>%
  select(person_id, question_code, marks) %>%
  pivot_wider(names_from = question_code, values_from = marks) %>%
  arrange(person_id)

X <- resp_wide %>% select(-person_id) %>% as.data.frame()
rownames(X) <- resp_wide$person_id

# Ensure only 0/1/NA
X[] <- lapply(X, function(col) ifelse(col %in% c(0, 1), col, NA))

# ---- detect items with only one response category ----
item_stats <- tibble(
  item = names(X),
  n_obs = sapply(X, function(z) sum(!is.na(z))),
  p_correct = sapply(X, function(z) mean(z, na.rm = TRUE)),
  n_cat = sapply(X, function(z) length(unique(na.omit(z))))
)

bad_items <- item_stats %>% filter(n_cat < 2)
good_items <- item_stats %>% filter(n_cat >= 2) %>% pull(item)

# Inspect which ones are "all correct" vs "all wrong"
bad_items <- bad_items %>%
  mutate(type = case_when(
    p_correct == 1 ~ "All correct (too easy)",
    p_correct == 0 ~ "All wrong (too hard)",
    TRUE ~ "Other"
  ))

print(bad_items)

# ---- fit Rasch only on estimable items ----
X2 <- X[, good_items, drop = FALSE]

mod_rasch <- mirt(
  data = X2,
  model = 1,
  itemtype = "Rasch",
  technical = list(NCYCLES = 800)
)

# ---- person abilities (EAP) + SE + 95% CI ----
theta <- fscores(mod_rasch, method = "EAP",
                 full.scores = TRUE, full.scores.SE = TRUE)

theta_df <- tibble(
  person_id = rownames(X2),
  ability = as.numeric(theta[, "F1"]),
  se = as.numeric(theta[, "SE_F1"]),
  lo = ability - 1.96 * se,
  hi = ability + 1.96 * se,
  model = sub("::run.*$", "", person_id)
)

model_abil <- theta_df %>%
  group_by(model) %>%
  summarise(
    ability = mean(ability),
    se = sd(ability) / sqrt(n()),
    lo = ability - 1.96 * se,
    hi = ability + 1.96 * se,
    .groups = "drop"
  ) %>%
  arrange(ability) %>%
  mutate(model = factor(model, levels = model))

# Ability plot (forest)
library(ggplot2)
ggplot(model_abil, aes(x = ability, y = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2) +
  geom_point(size = 2) +
  labs(x = "Rasch ability (theta; mean across runs) with 95% CI", y = NULL) +
  theme_bw()

# ---- item difficulties (b) ----
item_par <- coef(mod_rasch, IRTpars = TRUE, simplify = TRUE)$items %>%
  as.data.frame() %>%
  rownames_to_column("question_code")

item_diff <- item_par %>%
  transmute(question_code, b = b) %>%
  arrange(b)

# Item difficulty plot (estimable items only)
ggplot(item_diff, aes(x = b, y = reorder(question_code, b))) +
  geom_point(alpha = 0.8) +
  labs(x = "Item difficulty b (higher = harder)", y = NULL) +
  theme_bw() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  
  
  # Rasch ability plot with family overlay
  
  library(ggplot2)

abil_plot_df <- model_abil %>%
  mutate(
    family = add_family(as.character(model)),
    model  = as.character(model)
  ) %>%
  arrange(family, ability) %>%
  mutate(model = factor(model, levels = model))

ggplot(abil_plot_df, aes(x = ability, y = model, color = family)) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2) +
  geom_point(size = 2) +
  labs(x = "Rasch ability (theta; mean across runs) with 95% CI", y = NULL, color = "Family") +
  theme_bw()
  
  
  # Rasch ability GLMM correlation.
  
  emm <- readr::read_csv("emm_prob_df.csv") %>%
  transmute(
    model = as.character(model),
    glmm_acc = prob,
    glmm_lo = asymp.LCL,
    glmm_hi = asymp.UCL
  )

conv <- model_abil %>%
  mutate(model = as.character(model)) %>%
  inner_join(emm, by = "model") %>%
  mutate(family = add_family(model))

# Pearson and Spearman correlations
pear <- cor.test(conv$ability, conv$glmm_acc, method = "pearson")
spear <- cor.test(conv$ability, conv$glmm_acc, method = "spearman")

pear
spear



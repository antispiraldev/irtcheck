# Fit a 2PL in mirt and write the estimates out in both of mirt's own
# parameterisations.
#
#   Rscript crosscheck/fit_mirt.R <wide.csv> <out.csv>
#
# mirt is the psychometrics reference implementation: marginal maximum
# likelihood by EM with Gauss-Hermite quadrature over a fixed standard-normal
# ability distribution. That last part is what makes it comparable to us at
# all — it pins location and scale exactly the way fit/model.py's
# "Identification" docstring does — and unlike us it is a penalty-free
# maximiser, so where our estimates are shrunk and its are not, the difference
# is our prior.
#
# The output deliberately carries BOTH forms:
#
#   a1, d    mirt's internal slope-intercept form, P = 1/(1+exp(-(a1*theta+d)))
#   a, b     mirt's own IRTpars=TRUE conversion, P = 1/(1+exp(-a*(theta-b)))
#
# conventions.py converts a1/d to a/b itself with b = -d/a1, and compare.py
# checks that conversion against the a/b mirt reports. That is the single most
# valuable line in this directory: it proves our understanding of the
# slope-intercept trap is right, using mirt as its own witness, rather than
# asserting it in a comment.

suppressPackageStartupMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript fit_mirt.R <wide.csv> <out.csv>")
}
in_path <- args[1]
out_path <- args[2]

wide <- read.csv(in_path, check.names = FALSE, na.strings = "NA")
respondent_ids <- as.character(wide[["respondent_id"]])
resp <- wide[, setdiff(names(wide), "respondent_id"), drop = FALSE]
item_ids <- names(resp)

cat(sprintf(
  "mirt %s on R %s: %d respondents x %d items\n",
  as.character(packageVersion("mirt")), getRversion(),
  nrow(resp), ncol(resp)
))

# mirt refuses an item whose observed responses are all 0 or all 1, and it is
# right to: such a column carries no information about a slope at all, and the
# likelihood is maximised by sending `a` off to infinity with `b` chasing it.
# Our fitter does not refuse them — its prior keeps the estimate finite and it
# returns a number.
#
# This is a genuine behavioural difference between the two packages, not a
# nuisance, so it is recorded rather than quietly worked around: the dropped
# items are written out with NA parameters, and run_mirt.py propagates the NAs
# so the comparison restricts itself to the items mirt actually fitted and
# reports how many that was.
observed_levels <- sapply(resp, function(col) length(unique(col[!is.na(col)])))
degenerate <- item_ids[observed_levels < 2]
fitted_ids <- item_ids[observed_levels >= 2]
if (length(degenerate) > 0) {
  cat(sprintf(
    "dropping %d single-category item(s) mirt cannot estimate: %s\n",
    length(degenerate), paste(degenerate, collapse = ", ")
  ))
}
if (length(fitted_ids) < 3) {
  stop("fewer than 3 estimable items remain; nothing to compare")
}
resp <- resp[, fitted_ids, drop = FALSE]

# TOL is tightened from mirt's 1e-4 default and the iteration cap raised: this
# runs once by hand, so there is no reason to let the EM stop while it is still
# moving and then attribute the leftover motion to a convention difference.
fit <- mirt(
  data = resp,
  model = 1,
  itemtype = "2PL",
  method = "EM",
  SE = TRUE,
  technical = list(NCYCLES = 5000),
  TOL = 1e-6,
  verbose = FALSE
)

cat(sprintf(
  "converged=%s cycles=%d logLik=%.4f\n",
  extract.mirt(fit, "converged"), extract.mirt(fit, "iterations"),
  extract.mirt(fit, "logLik")
))

slope_intercept <- coef(fit, simplify = TRUE, IRTpars = FALSE)$items
irt_pars <- coef(fit, simplify = TRUE, IRTpars = TRUE)$items
item_ids <- fitted_ids

# Standard errors on the slope-intercept parameters, so compare.py can say
# whether a gap between mirt and us is inside mirt's own uncertainty. mirt
# returns these on the estimated (slope-intercept) scale, which is the scale
# they are meaningful on.
se <- coef(fit, printSE = TRUE, IRTpars = FALSE)
se_a1 <- sapply(item_ids, function(i) {
  block <- se[[i]]
  if (is.null(block) || !("a1" %in% colnames(block)) || !("SE" %in% rownames(block))) {
    NA_real_
  } else {
    block["SE", "a1"]
  }
})
se_d <- sapply(item_ids, function(i) {
  block <- se[[i]]
  if (is.null(block) || !("d" %in% colnames(block)) || !("SE" %in% rownames(block))) {
    NA_real_
  } else {
    block["SE", "d"]
  }
})

# EAP rather than ML ability estimates: ML diverges to +/-Inf for a respondent
# who got everything right or everything wrong, and the sparse dataset has
# those. EAP against the same standard normal that identifies the fit is both
# finite and the estimate on the metric we are comparing on.
theta <- fscores(fit, method = "EAP", full.scores = TRUE, full.scores.SE = FALSE)

items_out <- data.frame(
  item_id = item_ids,
  a1 = as.numeric(slope_intercept[, "a1"]),
  d = as.numeric(slope_intercept[, "d"]),
  mirt_a = as.numeric(irt_pars[, "a"]),
  mirt_b = as.numeric(irt_pars[, "b"]),
  se_a1 = as.numeric(se_a1),
  se_d = as.numeric(se_d),
  stringsAsFactors = FALSE
)
write.csv(items_out, out_path, row.names = FALSE, na = "NA")

theta_out <- data.frame(
  respondent_id = respondent_ids,
  theta = as.numeric(theta[, 1]),
  stringsAsFactors = FALSE
)
write.csv(theta_out, sub("\\.csv$", "-theta.csv", out_path), row.names = FALSE)

meta <- data.frame(
  key = c(
    "mirt_version", "r_version", "converged", "iterations", "logLik", "n_est_pars",
    "n_items_fitted", "n_items_dropped", "items_dropped"
  ),
  value = c(
    as.character(packageVersion("mirt")),
    as.character(getRversion()),
    as.character(extract.mirt(fit, "converged")),
    as.character(extract.mirt(fit, "iterations")),
    sprintf("%.6f", extract.mirt(fit, "logLik")),
    as.character(extract.mirt(fit, "nest")),
    as.character(length(fitted_ids)),
    as.character(length(degenerate)),
    paste(degenerate, collapse = " ")
  ),
  stringsAsFactors = FALSE
)
write.csv(meta, sub("\\.csv$", "-meta.csv", out_path), row.names = FALSE)

cat(sprintf("wrote %s\n", out_path))
